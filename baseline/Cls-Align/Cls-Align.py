             
  
                                                                         
  
                                                                              
                                                                               
                                                                              
                                                                           
                                                                       
                                                          
  
                                                                                
                                                 
  
                                                                            
                                                                          
                                                                             
                                                                        
                                                                               
                                                                               
           

import argparse

if __name__ == "__main__":
    _parser = argparse.ArgumentParser("Cls-Align baseline (public release)")
    _parser.add_argument("--config", required=True, help="Path to baseline config JSON.")
    _parser.parse_args()
    raise RuntimeError(
        "This baseline is intentionally disabled in the public release. "
        "Only core BrainMosaic-SID training is runnable."
    )

import os
import csv
import math
import json
import torch
import numpy
import random
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from eegcnn import EEGcnn, PositionalEncoding
from collections import defaultdict, Counter
from sklearn.metrics import f1_score
from eeg import custom_collate_fn, load_dataset
from config_args import getPath

class EEGclassification(torch.nn.Module):
    def __init__(self, chans=122, timestamp=165, cls=3, dropout1=0.1, dropout2=0.1, layer=0, 
                 pooling=None, size1=8, size2=8, feel1=125, feel2=25, emb_dim=256):
        super().__init__()
        self.eegcnn = EEGcnn(Chans=chans, kernLength1=feel1, kernLength2=feel2, F1=size1, D=size2, F2=size1*size2, P1=2, P2=5, dropoutRate=dropout1)
        self.hidden_dim = timestamp*size1*size2 if pooling is None else size1*size2
        self.linear = torch.nn.Linear(self.hidden_dim, cls)
        self.layer = layer
        self.pooling = pooling
        if self.layer > 0:
            self.poscode = PositionalEncoding(size1*size2, dropout=dropout2, max_len=timestamp)
            self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=size1*size2, nhead=size1*size2//8, dim_feedforward=4*size1*size2, batch_first=True, dropout=dropout2), num_layers=self.layer)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout2),
            torch.nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, return_features: bool=False):
                                                             
        hidden = self.eegcnn(inputs).permute(0, 2, 1)
        if self.layer > 0:        
            hidden = self.poscode(hidden)
            hidden = self.encoder(hidden, src_key_padding_mask=(mask.bool()==False))
        if self.pooling is None: hidden = torch.flatten(hidden, start_dim=1)
        if self.pooling == "mean": hidden = torch.sum(hidden*mask.unsqueeze(dim=2), dim=1)/torch.sum(mask, dim=1).unsqueeze(dim=1)
        if self.pooling == "sums": hidden = torch.sum(hidden*mask.unsqueeze(dim=2), dim=1)
        if self.pooling == "tops": hidden = hidden[:, 0, :]
        if return_features:
            projected_features = self.projection(hidden)
            return projected_features
        else:
            output = self.linear(hidden)
            return output
        
    def get_embeddings(self, inputs: torch.Tensor, mask: torch.Tensor):
        projected_features = self.forward(inputs, mask, return_features=True)
        return projected_features
    

def build_model(args, freeze_backbone=True):
    model = EEGclassification(
        chans=args.chans, timestamp=args.timestamp, 
        cls=args.cls, dropout1=args.dropout1, dropout2=args.dropout2, 
        layer=args.layer, pooling=args.pooling, size1=args.size1, size2=args.size2, 
        feel1=args.feel1, feel2=args.feel2, emb_dim=args.emb_dim)

    if not os.path.exists(args.model_path) or args.train_phase == 'train':
        if args.train_phase != 'train':
            raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
        return model
    
    state_dict = torch.load(args.model_path, map_location=args.device)
    filtered_state_dict = {}
    if args.train_phase == 'clip':
        for key, value in state_dict.items():
            if not key.startswith('projection.') and not key.startswith('classifier.'):
                filtered_state_dict[key] = value
    else:
        for key, value in state_dict.items():
            filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith('projection.'):
                param.requires_grad = False
    
    model.to(device)
    return model





def valid(valid_dataloader, device, model):
    with torch.no_grad():
        model.eval()
        valid_output = []
        valid_target = []
        for idy, (valid_input_features, valid_labels, valid_length) in enumerate(valid_dataloader):
            valid_input_features['eeg'] = valid_input_features['eeg'].to(device)
            valid_labels = valid_labels.to(device)
            valid_length = valid_length.to(device)
                                                                                             
                                                                           
            valid_output.append(model(valid_input_features['eeg'], valid_length))
            valid_target.append(valid_labels)
        valid_output = torch.cat(valid_output, dim=0)
        valid_target = torch.cat(valid_target, dim=0)
        valid_ans = torch.max(valid_output, dim=1)[1]
        print(valid_output.shape, valid_target.shape)
        valid_accu = (valid_ans == valid_target).float().mean()
        valid_maf1 = f1_score(valid_target.tolist(), valid_ans.tolist(), average='macro')
    return valid_output, valid_target, valid_accu, valid_maf1, valid_ans
    

def train(device, train_dataloader, valid_dataloader, model, config, label_frequency):
    model.to(device)
    label_frequency = torch.log(label_frequency.pow(config.tau)+1e-12).unsqueeze(dim=0)
    label_frequency = label_frequency.to(device)
    print(label_frequency.dtype, label_frequency.shape, label_frequency)

    training_step = len(train_dataloader)*config.epoch
    warmup_step = math.ceil(training_step*config.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr1, weight_decay=config.wd1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, training_step)

    running_loss = 0.0
    max_accuracy = 0.0  
    max_f1scores = 0.0
    for epoch in range(config.epoch):
        for idx, (input_features, labels, length) in enumerate(train_dataloader):
            step = epoch*len(train_dataloader)+idx+1
            model.train()

            input_features['eeg'] = input_features['eeg'].to(device)
            labels = labels.to(device)
            length = length.to(device)
                                                                                 
                                                               

            optimizer.zero_grad()
            output = model(input_features['eeg'], length)
            loss = torch.nn.functional.cross_entropy(output+label_frequency, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            
        print("step:{}(epoch{} {}/{}) loss:{}".format(step, epoch, idx, len(train_dataloader), running_loss/config.train_log))
        running_loss = 0.0
        valid_output, valid_target, valid_accu, valid_maf1, valid_ans = valid(valid_dataloader, device, model)
        valid_loss = torch.nn.functional.cross_entropy(valid_output+label_frequency, valid_target)
        print("step:{}(epoch{} {}/{}) valid_loss:{} accuracy:{} accuracy:{} f1:{} max_f1:{}".format(step, epoch, idx, len(train_dataloader), valid_loss.item(), valid_accu.item(), max_accuracy, valid_maf1, max_f1scores))

        if valid_accu > max_accuracy:
            max_accuracy = valid_accu
            print("saving model at epoch="+str(epoch)+"..."+f" The accuracy: {valid_accu:.05f}")
            label_path = os.path.join(args.checkpoint_path, f'best_label_cls_{args.task}_{args.sub}.json')
                
            torch.save(model.state_dict(), args.model_path)
            label_dict = {'ground_truth': [i.item() for i in valid_target], 
                          'predict': [i.item() for i in valid_ans]}
            with open(label_path, 'w') as f:
                json.dump(label_dict, f, indent=4, ensure_ascii=False)         

    print("result:", max_accuracy)


def compute_loss(args, eeg_embeddings, text_embeddings):
    cosine_loss = torch.nn.CosineEmbeddingLoss()
    target = torch.ones(eeg_embeddings.size(0)).to(args.device)
    cosine_loss = cosine_loss(eeg_embeddings, text_embeddings, target)

    batch_size = eeg_embeddings.size(0)
    labels = torch.arange(batch_size).to(args.device)
    eeg_norm = F.normalize(eeg_embeddings, p=2, dim=1)
    text_norm = F.normalize(text_embeddings, p=2, dim=1)
    
    similarity_matrix = torch.matmul(eeg_norm, text_norm.t()) / args.temperature
    info_nce_loss = (F.cross_entropy(similarity_matrix, labels) + 
                    F.cross_entropy(similarity_matrix.t(), labels)) / 2
    
    total_loss = cosine_loss + info_nce_loss
    
    return total_loss, {
        'cosine_loss': cosine_loss.item(),
        'info_nce_loss': info_nce_loss.item(),
        'total_loss': total_loss.item()
    }
    
    
def compute_metrics(args, eeg_embeddings, text_embeddings):
    batch_size = eeg_embeddings.size(0)
    eeg_norm = F.normalize(eeg_embeddings, p=2, dim=1)
    text_norm = F.normalize(text_embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(eeg_norm, text_norm.t())
    
                      
    avg_similarity = torch.diag(similarity_matrix).mean()
    mask = ~torch.eye(batch_size, dtype=torch.bool).to(args.device)
    neg_avg_similarity = similarity_matrix[mask].mean()
    
    return {
        'avg_similarity': avg_similarity.item(),
        'neg_avg_similarity': neg_avg_similarity.item(),
    }


def valid_alignment(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    avg_similarity, neg_avg_similarity = 0.0, 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            eeg_data, text_data, eeg_mask, text_emb = batch
            eeg_data, eeg_mask, text_emb = eeg_data.to(device), eeg_mask.to(device), text_emb.to(device)

            eeg_embeddings = model.get_embeddings(eeg_data, eeg_mask)
            loss, loss_dict = compute_loss(args, eeg_embeddings, text_emb)
            metrics = compute_metrics(args, eeg_embeddings, text_emb)
            
            val_loss += loss_dict['total_loss']
            avg_similarity += metrics['avg_similarity']
            neg_avg_similarity += metrics['neg_avg_similarity']
            
    avg_val_loss = val_loss / len(val_loader)
    avg_similarity = avg_similarity / len(val_loader)
    neg_avg_similarity /= len(val_loader)
    
    return avg_val_loss, avg_similarity, neg_avg_similarity


def train_alignment(args, model, train_loader, val_loader):    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr1, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        avg_similarity = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            eeg_data, text_data, eeg_mask, text_emb = batch
            eeg_data, eeg_mask, text_emb = eeg_data.to(device), eeg_mask.to(device), text_emb.to(device)

            optimizer.zero_grad()
            eeg_embeddings = model.get_embeddings(eeg_data, eeg_mask)
            loss, loss_dict = compute_loss(args, eeg_embeddings, text_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            metrics = compute_metrics(args, eeg_embeddings, text_emb)
            
                  
            train_loss += loss_dict['total_loss']
            avg_similarity += metrics['avg_similarity']
        
                  
        avg_train_loss = train_loss / len(train_loader)
        avg_train_sim = avg_similarity / len(train_loader)
        print(f'Epoch: {epoch}, Train loss: {avg_train_loss}, Train sim: {avg_train_sim}')
        
        val_loss, avg_similarity, neg_avg_similarity = valid_alignment(model, val_loader, device)
        print(f'Valid loss: {val_loss}, Valid sim: {avg_similarity}, Valid neg sim: {neg_avg_similarity}')
        scheduler.step(val_loss)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            print(f'Save model in epoch {epoch}')
            torch.save(model.state_dict(), args.model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
    return model


def init_csv_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'mean_cosine', 'neg_cosine'])

def append_to_csv(file_path, dataset_name, mean_cosine, neg_cosine):
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([dataset_name, mean_cosine, neg_cosine])
        

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu_id', type=int, default=2)
parser.add_argument('--lr1', type=float, default=5e-4)        
parser.add_argument('--wd1', type=float, default=0.01)
parser.add_argument('--tau', type=float, default=0.0)
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--train_log', type=int, default=10)
parser.add_argument('--evals_log', type=int, default=100)
parser.add_argument('--checkpoint_log', type=int, default=100)
parser.add_argument('--chans', type=int, default=122)
parser.add_argument('--timestamp', type=int, default=250)        
parser.add_argument('--pooling', type=str, default='mean')        
parser.add_argument('--size1', type=int, default=16)
parser.add_argument('--size2', type=int, default=16)
parser.add_argument('--feel1', type=int, default=20)         
parser.add_argument('--feel2', type=int, default=10)        
parser.add_argument('--emb_dim', type=int, default=256)         
parser.add_argument('--cls', type=int, default=3)
parser.add_argument('--layer', type=int, default=1)
parser.add_argument('--dropout1', type=float, default=0.5)       
parser.add_argument('--dropout2', type=float, default=0.5)       
parser.add_argument('--temperature', type=float, default=0.07)       
parser.add_argument('--sub', type=str, default='ZJS')
parser.add_argument('--rand_guess', type=int, default=0)                                                                                           
parser.add_argument('--dataset', type=str, default='Zuco_1')
parser.add_argument('--task', type=str, default='task1-SR')
parser.add_argument('--subset_ratio', type=float, default=1.0)
parser.add_argument('--train_phase', type=str, default='clip')       
args = parser.parse_args()
args = getPath(args)
print(args)

seed = args.seed
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count() else "cpu")
args.device = device
args.model_path = os.path.join(args.checkpoint_path, f'checkpoint_best_cls_{args.task}_{args.sub}.pt')
model = build_model(args, freeze_backbone=True)

trainset, validset, trainloader, validloader = load_dataset(args)
print(len(trainset), len(trainloader))
print(len(validset), len(validloader))    

if not os.path.exists(args.checkpoint_path): os.mkdir(args.checkpoint_path)
if args.train_phase == 'train':
    label_freqs = [0.0 for idx in range(args.cls)]
    label_count = Counter(trainset.labels)
    for i in label_count: label_freqs[i] = label_count[i]/len(trainset)
    label_freqs = torch.tensor(label_freqs)
    print(label_freqs)
    train(device, trainloader, validloader, model, args, label_frequency=label_freqs)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)
    _, _, valid_accu, valid_maf1, valid_ans = valid(validloader, device, model)
    if args.dataset == 'Chisco':
        with open(args.label_path, 'r', encoding='gbk') as f:
            class_num = json.load(f)
        label_path = os.path.join(args.checkpoint_path, f'best_label_cls_{args.task}_{args.sub}.json')
        with open(label_path, 'r') as f:
            ground_pred = json.load(f)
        ground_pred['ground_text'] = [class_num[str(i)] for i in ground_pred['ground_truth']]
        ground_pred['pred_text'] = [class_num[str(i)] for i in ground_pred['predict']]
        with open(label_path, 'w') as f:
            json.dump(ground_pred, f, indent=4, ensure_ascii=False)
    
elif args.train_phase == 'clip':
    args.model_path = os.path.join(args.checkpoint_path, f'checkpoint_best_cls_{args.task}_{args.sub}_clip.pt')
    trained_model = train_alignment(args, model, trainloader, validloader)
    avg_val_loss, avg_similarity, neg_avg_similarity = valid_alignment(model, validloader, device)
    if not os.path.exists('ans_cls.csv'):
        init_csv_file('ans_cls.csv')
    dataset_name = f"{args.dataset}_{args.sub}_{args.task}"
    append_to_csv('ans_cls.csv', dataset_name, avg_similarity, neg_avg_similarity)
