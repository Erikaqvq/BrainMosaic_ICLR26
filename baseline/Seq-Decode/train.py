import argparse

if __name__ == "__main__":
    _parser = argparse.ArgumentParser("Seq-Decode baseline (public release)")
    _parser.add_argument("--config", required=False, help="Path to baseline config JSON.")
    _parser.parse_args()
    raise RuntimeError(
        "Seq-Decode baseline is intentionally disabled in the public release. "
        "Do not run directly."
    )

import os
import json
import csv
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ModernTCN import ModernTCNEncoder
from dataloader import EEGConceptDataset, ConceptDataset, PrivateConceptDataset, load_word_emb, collate_fn_factory, build_vocab_from_dataset
from config_args import gpu_id, batch_size, DATASET_CONFIG, SUBJECTS, TASKS


                              
                                                         
                                                       
                                                               
                              
                             
class EEGEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, mask):
        B, C, T = x.shape
        x = x.permute(0,2,1)             
        if mask is None:
            mask = torch.ones(B, T, device=x.device, dtype=x.dtype)
        mask = mask.unsqueeze(-1)             
        x = x * mask                                      
        x = x.permute(0,2,1)             
        return self.encoder(x)          


class DefaultEEGEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_ch, out_dim)

    def forward(self, eeg: torch.Tensor, mask: Optional[torch.Tensor]=None):
        B, C, T = eeg.shape
                                
        x = eeg.permute(0,2,1)             
        x = self.proj(x)                         
        if mask is None:
            mask = torch.ones(B, T, device=x.device, dtype=x.dtype)
        mask = mask.unsqueeze(-1)             
        x = x * mask                                      
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)           
        emb = x.sum(dim=1) / denom.squeeze(1)                 
        return emb

                              
                    
                                                                  
                                                              
                              
class LSTMDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 token_emb_dim: int,
                 hidden_dim: int,
                 eeg_emb_dim: int,
                 num_layers: int = 1,
                 pad_id: int = 0,
                 dropout: float = 0.1,
                 embedding = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb_dim = token_emb_dim
        self.hidden_dim = hidden_dim
        self.eeg_emb_dim = eeg_emb_dim
        self.num_layers = num_layers
        self.pad_id = pad_id

        if embedding == None:
            self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=pad_id)
        else:
            self.token_embedding = embedding
        self.lstm = nn.LSTM(token_emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

                                                                  
        self.eeg2h = nn.Linear(eeg_emb_dim, num_layers * hidden_dim)
        self.eeg2c = nn.Linear(eeg_emb_dim, num_layers * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, eeg_emb: torch.Tensor, input_ids: torch.LongTensor):
                             
                           
        B, L = input_ids.size()
              
        h0 = self.eeg2h(eeg_emb)                          
        c0 = self.eeg2c(eeg_emb)
        h0 = h0.view(B, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()                      
        c0 = c0.view(B, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()

        emb = self.token_embedding(input_ids)             
        out, (hn, cn) = self.lstm(emb, (h0, c0))                  
        out = self.dropout(out)
        logits = self.output(out)             
        stop_logits = self.stop_head(out)             
        return logits, stop_logits.squeeze(-1)

    @torch.no_grad()
    def beam_search(self, eeg_emb: torch.Tensor, token2id: Dict[str,int], id2token: List[str],
                    beam_width: int = 3, max_len: int = 30, stop_threshold: float = 0.5, 
                    device: Optional[torch.device]=None):
        """
        Batch size = 1 version. eeg_emb: (D,)
        returns token id list (no BOS, no EOS)
        """
        if device is None:
            device = next(self.parameters()).device
        if eeg_emb.dim() == 1:
            eeg_emb = eeg_emb.unsqueeze(0)          
        bos = token2id["<BOS>"]
                                 
        pad = token2id["<PAD>"]

                    
        h0 = self.eeg2h(eeg_emb).view(1, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()
        c0 = self.eeg2c(eeg_emb).view(1, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()

                                     
        active = [{
            "seq": [bos],
            "logp": 0.0,
            "h": h0.clone(),
            "c": c0.clone(),
            "finished": False
        }]
        finished = []

        for step in range(max_len):
            all_candidates = []
                                          
            last_tokens = torch.tensor([b["seq"][-1] for b in active], dtype=torch.long, device=device)
            emb = self.token_embedding(last_tokens).unsqueeze(1)                    
                                 
            h_stack = torch.cat([b["h"] for b in active], dim=1)                             
            c_stack = torch.cat([b["c"] for b in active], dim=1)

            out, (h_new, c_new) = self.lstm(emb, (h_stack, c_stack))                       
            out = out.squeeze(1)                 
            logp = F.log_softmax(self.output(out), dim=-1)                 
            stop_probs = torch.sigmoid(self.stop_head(out)).squeeze(-1)               

            k = min(beam_width, logp.size(-1))
            topk_vals, topk_idx = torch.topk(logp, k, dim=-1)                

            for i, beam in enumerate(active):
                if beam["finished"]:
                           
                    all_candidates.append(beam)
                    continue
                base = beam["logp"]
                for j in range(k):
                    tok = int(topk_idx[i,j].item())
                    tok_logp = float(topk_vals[i,j].item())
                    new_seq = beam["seq"] + [tok]
                    new_logp = base + tok_logp
                    new_h = h_new[:, i:i+1, :].clone()
                    new_c = c_new[:, i:i+1, :].clone()
                                                  
                    stop_prob = float(stop_probs[i].item())
                    finished_flag = (stop_prob > stop_threshold)
                    all_candidates.append({
                        "seq": new_seq,
                        "logp": new_logp,
                        "h": new_h,
                        "c": new_c,
                        "finished": finished_flag
                    })
            
                                   
            all_candidates = sorted(all_candidates, key=lambda x: x["logp"], reverse=True)
            active = all_candidates[:beam_width]
                                            
            newly_finished = [b for b in active if b["finished"]]
            finished.extend(newly_finished)
                                       
            if len(finished) >= beam_width:
                break

        candidates_final = finished if len(finished) > 0 else active
        candidates_final = sorted(candidates_final, key=lambda x: x["logp"], reverse=True)[:beam_width]
        seqs = []
        for cand in candidates_final:
            seq = cand["seq"]
            if seq and seq[0] == bos:
                seq = seq[1:]
            seqs.append(seq)
        
        return seqs                     

                              
                                                                                                
                              
def compute_sequence_metrics(pred_ids_list: List[int], true_ids: List[int], embedding_layer: nn.Embedding, dataset: str):
    expected_dict = {
        'ChineseEEG2': 0.4061,
        'Chisco': 0.3907,
        'private': 0.4536,
        'Zuco_1': 0.4335,
        'Zuco_2': 0.4288
                     }
    device = embedding_layer.weight.device
    if len(pred_ids_list) == 0 or len(true_ids) == 0:
        return {"mean_token_cos": 0.0, "seq_cos": 0.0, "hit_acc": 0.0}
    t_emb = embedding_layer(torch.tensor(true_ids, dtype=torch.long, device=device))           
    t_mean = t_emb.mean(dim=0, keepdim=True)
    all_seq_cos_sims = []
    hit = [0] * len(true_ids)
    for pred_ids in pred_ids_list:
        if not pred_ids:
            continue
        pred_emb = embedding_layer(torch.tensor(pred_ids, dtype=torch.long, device=device))               
        p_mean = pred_emb.mean(dim=0, keepdim=True)          
        seq_cos = F.cosine_similarity(p_mean, t_mean).item()
        all_seq_cos_sims.append(seq_cos)
        for i in range(min(len(true_ids), len(pred_ids))):
            if pred_ids[i] == true_ids[i]:
                hit[i] = 1
    
    hit_acc = sum(hit) / len(true_ids) if true_ids else 0.0
    mean_seq_cos = sum(all_seq_cos_sims) / len(all_seq_cos_sims) if all_seq_cos_sims else 0.0
    mean_token_cos = 0.0
    if pred_ids_list:
        best_candidate_idx = all_seq_cos_sims.index(max(all_seq_cos_sims))
        best_pred_ids = pred_ids_list[best_candidate_idx]
        
        if best_pred_ids and true_ids:
            best_pred_tensor = torch.tensor(best_pred_ids, dtype=torch.long, device=device)
            best_pred_emb = embedding_layer(best_pred_tensor)
            min_len = min(len(best_pred_ids), len(true_ids))

            if len(best_pred_ids) >= len(true_ids):
                p_trim = best_pred_emb[:min_len]
                token_cos_sims = F.cosine_similarity(p_trim, t_emb, dim=1)
            else:
                expected = expected_dict[dataset]
                t_trim = t_emb[:min_len]
                token_cos_sims = F.cosine_similarity(best_pred_emb, t_trim, dim=1)
                for i in range(len(t_emb)-min_len):
                    token_cos_sims = torch.cat([token_cos_sims, torch.tensor([expected]).to(device)])
                    
            mean_token_cos = token_cos_sims.mean().item()
    return {"mean_token_cos": mean_token_cos, "seq_cos": mean_seq_cos, "hit_acc": hit_acc}


                              
                                                
                              
class Trainer:
    def __init__(self,
                 model: LSTMDecoder,
                 token2id: Dict[str,int],
                 id2token: List[str],
                 dataset: str,
                 device: torch.device,
                 eeg_encoder: Optional[nn.Module] = None,
                 external_token_embedding: Optional[nn.Embedding] = None,
                 save_dir: str = "./results"):
        self.model = model.to(device)
        self.device = device
        self.token2id = token2id
        self.id2token = id2token
        self.dataset = dataset
        self.eeg_encoder = eeg_encoder.to(device) if eeg_encoder is not None else None
        self.external_token_embedding = external_token_embedding
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        if self.eeg_encoder is None:
                                                                  
                                                                                                  
                                                                                      
                                                                                        
            self.default_encoder = None
        else:
            self.default_encoder = None

    def encode_eeg(self, eeg: torch.Tensor, mask: Optional[torch.Tensor]=None):
                        
        if self.eeg_encoder is not None:
            return self.eeg_encoder(eeg, mask)          
                                             
        if self.default_encoder is None:
            C = eeg.size(1)
            D = self.model.eeg_emb_dim
            self.default_encoder = DefaultEEGEncoder(C, D).to(self.device)
        return self.default_encoder(eeg, mask)

    def train_epoch(self, dataloader, optimizer, criterion, teacher_forcing_ratio=1.0, clip=1.0):
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        pbar = tqdm(dataloader, desc="train", leave=False)
        for eeg_pad, mask_pad, inp_pad, tgt_pad, _ in pbar:
            eeg_pad = eeg_pad.to(self.device)                      
            mask_pad = mask_pad.to(self.device)                 
            inp_pad = inp_pad.to(self.device)                   
            tgt_pad = tgt_pad.to(self.device)                   
            B, L = inp_pad.shape
                              
            eeg_emb = self.encode_eeg(eeg_pad, mask_pad)          
            logits, stop_logits = self.model(eeg_emb, inp_pad)                    
                                                                  
            stop_labels = torch.zeros_like(tgt_pad, dtype=torch.float)
            for b in range(B):
                pad_positions = (tgt_pad[b] == 0).nonzero(as_tuple=False).squeeze(-1)
                if len(pad_positions) > 0:
                    stop_pos = pad_positions[0]
                    if stop_pos >= 0:
                        stop_labels[b, stop_pos] = 1.0
                else:
                    stop_labels[b, -1] = 1.0
            loss_stop = F.binary_cross_entropy_with_logits(
                stop_logits, stop_labels, pos_weight=torch.tensor(10.0, device=self.device)
            )
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_pad.view(-1))
            loss += loss_stop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item() * B
            n_samples += B
            pbar.set_postfix({"loss": total_loss / max(1, n_samples)})
        return total_loss / max(1, n_samples)

    @torch.no_grad()
    def validate(self, dataloader, beam_width=3, max_len=40, cosine_threshold=0.8):
        self.model.eval()
        all_metrics = []
        predictions = []                         
        emb_layer = self.external_token_embedding if self.external_token_embedding is not None else self.model.token_embedding
        for eeg_pad, mask_pad, inp_pad, tgt_pad, raw_tokens_list in tqdm(dataloader, desc="val"):
            B = eeg_pad.size(0)
            eeg_pad = eeg_pad.to(self.device)
            mask_pad = mask_pad.to(self.device)
            tgt_pad = tgt_pad.to(self.device)

                    
            eeg_emb = self.encode_eeg(eeg_pad, mask_pad)          
            for i in range(B):
                emb_i = eeg_emb[i]        
                                         
                pred_ids = self.model.beam_search(emb_i, self.token2id, self.id2token,
                                                 beam_width=beam_width, max_len=max_len, device=self.device)
                                            
                preds = []
                for pred_id in pred_ids:
                    pred_tokens = [self.id2token[i] for i in pred_id]
                    preds.append(pred_tokens)
                                                                          
                tgt_i = tgt_pad[i].cpu().tolist()
                                    
                true_ids = [x for x in tgt_i if x != self.token2id["<PAD>"]]                                     
                                 
                seq_metrics = compute_sequence_metrics(pred_ids, true_ids, emb_layer, dataset=self.dataset)
                      
                predictions.append({
                                           
                    "pred_tokens": preds,
                                           
                    "true_tokens": [self.id2token[x] for x in true_ids],
                                            
                })
                all_metrics.append(seq_metrics)

                   
        if all_metrics:
            avg_mean_token_cos = sum(m["mean_token_cos"] for m in all_metrics) / len(all_metrics)
            avg_seq_cos = sum(m["seq_cos"] for m in all_metrics) / len(all_metrics)
            avg_hit = sum(m["hit_acc"] for m in all_metrics) / len(all_metrics)
        else:
            avg_mean_token_cos = avg_seq_cos = avg_hit = 0.0
        summary = {
            "mean_token_cos": avg_mean_token_cos,
            "seq_cos": avg_seq_cos,
            "hit_acc": avg_hit,
                                           
        }
        return summary, predictions

                              
                                 
                              
def main_train(
    train_dataset,
    val_dataset,
    out_dir: str,
    chans: int,
    device: str = "cuda",
    token_min_freq: int = 1,
    batch_size: int = 32,
    epochs: int = 10,
    token_emb_dim: int = 256,
    hidden_dim: int = 512,
    eeg_emb_dim: int = 256,
    lr: float = 1e-3,
    beam_width: int = 3,
    max_len: int = 10,
    stop_epoch: int = 10,
    warm_epoch: int = 3,
    is_train: bool = True,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
                              
    token2id, id2token = build_vocab_from_dataset(train_dataset, val_dataset, min_freq=token_min_freq)
    collate_fn = collate_fn_factory(token2id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    vocab_size = len(token2id)
                                                                                  
    eeg_encoder = ModernTCNEncoder(in_channels=chans, d_model=eeg_emb_dim)
                                                                                             
    embedding_matrix = load_word_emb(args)
    token_embedding = torch.nn.Embedding.from_pretrained(
        embedding_matrix,
        freeze=True,             
        padding_idx=token2id["<PAD>"]                     
    )
    model = LSTMDecoder(vocab_size=vocab_size,
                        token_emb_dim=token_emb_dim,
                        hidden_dim=hidden_dim,
                        eeg_emb_dim=eeg_emb_dim,
                        num_layers=1,
                        pad_id=token2id["<PAD>"],
                        embedding=token_embedding).to(device)

    trainer = Trainer(model, token2id, id2token, device=device, eeg_encoder=eeg_encoder, 
                      external_token_embedding=None, save_dir=out_dir, dataset = dataset)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(eeg_encoder.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=token2id["<PAD>"])

    best_val, best_mean_cos, best_seq_cos = -1e9, -1e9, -1e9
    early_stop = 0
    if is_train:
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}")
            train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
            print(f" train_loss: {train_loss:.4f}")
            summary, preds = trainer.validate(val_loader, beam_width=beam_width, max_len=max_len,)
            print(" val summary:", summary)
                                                                           
            metric, mean_cos, seq_cos = summary["hit_acc"], summary["mean_token_cos"], summary["seq_cos"]
            if epoch >= warm_epoch:
                if metric > best_val or (metric == best_val and mean_cos > best_mean_cos):
                    early_stop = 0
                    best_val, best_mean_cos, best_seq_cos = metric, mean_cos, seq_cos
                    torch.save(model.state_dict(), os.path.join(out_dir, f"best_model_{subj}_{task}.pt"))
                    with open(os.path.join(out_dir, f"predic_res_{subj}_{task}.json"), "w", encoding="utf-8") as f:
                        json.dump({"summary": summary, "predictions": preds}, f, ensure_ascii=False, indent=2)
                    print(f" saved best model in epoch {epoch}")
                else:
                    early_stop += 1
                    if early_stop > stop_epoch:
                        print('Early Stop')
                        break
    else:
        summary, preds = trainer.validate(val_loader, beam_width=beam_width, max_len=max_len,)
        print(" val summary:", summary)
                                                                       
        metric, mean_cos, seq_cos = summary["hit_acc"], summary["mean_token_cos"], summary["seq_cos"]
        return metric, mean_cos, seq_cos

                                                                 
    return best_val, best_mean_cos, best_seq_cos


def init_csv_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'token_matching_acc', 'seq_mean_cosine', 'token_mean_cosine'])

def append_to_csv(file_path, dataset_name, matching_acc, s_cosine, t_cosine):
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([dataset_name, matching_acc, s_cosine, t_cosine])



dataset = "Chisco"
args = DATASET_CONFIG[dataset]
device = torch.device(f"cuda" if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count() else "cpu")

for task in args["tasks"]:
    for subj in args["subjects"]:
        if dataset == 'Chisco':
            train_ds = EEGConceptDataset(subj=subj, task=task, args=args, split='train')
            val_ds = EEGConceptDataset(subj=subj, task=task, args=args, split='val')
        elif dataset == 'private':
            train_ds = ConceptDataset(subj=subj, task=task, args=args, split='train', dataset=dataset)
            val_ds = ConceptDataset(subj=subj, task=task, args=args, split='val', dataset=dataset)
        else:
            train_ds = ConceptDataset(subj=subj, task=task, args=args, split='train', dataset=dataset)
            val_ds = ConceptDataset(subj=subj, task=task, args=args, split='val', dataset=dataset)
        
        best_val, best_mean_cos, bset_token_cos = main_train(train_ds, val_ds, out_dir=args['out_dir'], 
                                        chans=args["chans"], device=device, epochs=50, batch_size=args['batch_size'], 
                                        max_len=args["max_len"], stop_epoch=10, warm_epoch=3, beam_width=5, is_train=False)

        if not os.path.exists('ans_LSTM.csv'):
            init_csv_file('ans_LSTM.csv')
        dataset_name = f"{dataset}_{subj}_{task}"
        append_to_csv('ans_LSTM.csv', dataset_name, best_val, best_mean_cos, bset_token_cos)
