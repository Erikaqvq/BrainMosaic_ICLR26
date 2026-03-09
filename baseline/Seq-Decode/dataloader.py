import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import os
import json
import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Any


FILTER_POS_INITIALS = set("cpqur")
FILTER_POS_INITIALS_EN = ["AUX", "CCONJ", "ADP"]

def normalize_segmentation_columns(seg_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    cn_sentence = "\u53e5\u5b50"
    cn_token_prefix = "\u5206\u8bcd"
    if cn_sentence in seg_df.columns and "sentence" not in seg_df.columns:
        rename_map[cn_sentence] = "sentence"
    for c in seg_df.columns:
        cs = str(c)
        if cs.startswith(cn_token_prefix):
            rename_map[c] = cs.replace(cn_token_prefix, "token", 1)
    if rename_map:
        seg_df = seg_df.rename(columns=rename_map)
    return seg_df

def filter_word(info_path, dataset, row, token_cols):
    toks = [str(row[c]).strip() for c in token_cols if pd.notna(row[c]) and str(row[c]).strip()]
    filter_list = FILTER_POS_INITIALS
    if 'Zuco' in dataset:
        filter_list = FILTER_POS_INITIALS_EN
    with open(os.path.join(info_path, f'{dataset}_words_pos.json'), 'r') as f:
        total_data = json.load(f)
    word2pos = {item['key']: item['pos'] for item in total_data}
    toks = [tok for tok in toks if tok in word2pos and word2pos[tok] not in filter_list]
    return toks

                              
                                      
                                
                                                                                                      
                              
def collate_fn_factory(token2id: Dict[str,int]):
    pad_id = token2id["<PAD>"]
    bos_id = token2id["<BOS>"]
                                
    unk_id = token2id["<UNK>"]

    def collate_fn(batch: List[Dict[str,Any]]):
        eegs, masks, inp_ids_list, tgt_ids_list, raw_tokens_list = [], [], [], [], []
        for sample in batch:
            eeg = sample["eeg"]                                      
            mask = sample.get("mask", None)
            toks = sample["tokens"]                 
            if mask is None:
                                    
                mask = torch.ones(eeg.shape[-1], dtype=torch.float32)
                            
            inp = ["<BOS>"] + toks
                                    
            tgt = toks + ["<PAD>"]
            inp_ids = [token2id.get(t, unk_id) for t in inp]
            tgt_ids = [token2id.get(t, unk_id) for t in tgt]

            eegs.append(eeg)
            masks.append(mask)
            inp_ids_list.append(torch.tensor(inp_ids, dtype=torch.long))
            tgt_ids_list.append(torch.tensor(tgt_ids, dtype=torch.long))
            raw_tokens_list.append(toks)

                             
        inp_pad = pad_sequence(inp_ids_list, batch_first=True, padding_value=pad_id)          
        tgt_pad = pad_sequence(tgt_ids_list, batch_first=True, padding_value=pad_id)          
        eegs = torch.stack(eegs)
        masks = torch.stack(masks)

        return eegs, masks, inp_pad, tgt_pad, raw_tokens_list

    return collate_fn

class EEGConceptDataset(Dataset):
    def __init__(self, subj, task, args, split="train"):
        assert split in ("train", "val")
        self.split = split
        self.subj = subj
        self.task = task
        
                
        seg_df = pd.read_csv(args['segmentation_path'])
        seg_df = normalize_segmentation_columns(seg_df)
        seg_df = seg_df.dropna(subset=["sentence"])
        seg_df["sentence"] = seg_df["sentence"].astype(str).str.strip()
        with open(args['cluster_emb_path'], 'r') as f:
            self.word_cluster = json.load(f)

        token_cols = [c for c in seg_df.columns if str(c).startswith("token")]
        token_lookup = {}
        for _, row in seg_df.iterrows():
            sent = row["sentence"]
            toks = filter_word(args['filter_path'], 'Chisco', row, token_cols)
            token_lookup[sent] = toks

        valid_run_ids = set([f"0{i}" for i in range(1, 46)])
        full_data = [] 
        subj_path = os.path.join(args['eeg_root'], f"sub-{subj}", "eeg")
        if not os.path.exists(subj_path):
            raise FileNotFoundError(f"Subject folder not found: {subj_path}")

        for fn in sorted(os.listdir(subj_path)):
            if not (fn.endswith(".pkl") and f"sub-{subj}" in fn and f"task-{self.task}" in fn):
                continue
            try:
                run_id = fn.split("run-")[1].split("_")[0]
            except IndexError:
                continue
            if run_id not in valid_run_ids:
                continue

            with open(os.path.join(subj_path, fn), "rb") as f:
                trials = pickle.load(f)
            length = len(trials)
            train_len = int(length * 0.8)
            if split == "train":
                trials = trials[:train_len]
            else:
                trials = trials[train_len:]
            
            for tr in trials:
                sentence = str(tr.get("text", "")).strip()
                if sentence not in token_lookup:           
                    continue
                raw_tokens = token_lookup[sentence]
                eeg = tr["input_features"][0, :122, :].astype(np.float32) * 1e6            
                mask = np.ones(eeg.shape[-1], dtype=np.float32)                  
                full_data.append((eeg, mask, raw_tokens))

        if not full_data:
            raise ValueError(f"No valid trials found for subject {self.subj}, split={split}")

        self.data = []
        for eeg, mask, toks in full_data:
            cluster_toks = [self.word_cluster[i] for i in toks]
            self.data.append({
                "eeg": torch.from_numpy(eeg),
                "tokens": cluster_toks,
                "mask": torch.from_numpy(mask)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ConceptDataset(Dataset):
    def __init__(self, subj, task, args, split="train", dataset="ChineseEEG2"):
        assert split in ("train", "val")
        self.split = split
        self.subj = subj
        self.task = task
        with open(args['cluster_emb_path'], 'r') as f:
            self.word_cluster = json.load(f)

                                     
        encodings = ['utf-8', 'gbk']
        for encoding in encodings:
            try:
                seg_df = pd.read_csv(args['segmentation_path'], sep=',', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        seg_df = normalize_segmentation_columns(seg_df)
        seg_df = seg_df.dropna(subset=["sentence"])
        seg_df["sentence"] = seg_df["sentence"].astype(str).str.strip()

        token_cols = [c for c in seg_df.columns if str(c).startswith("token")]
        token_lookup = {}
        for _, row in seg_df.iterrows():
            sent = row["sentence"]
            toks = filter_word(args['filter_path'], dataset, row, token_cols)
            token_lookup[sent] = toks 

                    
        if self.subj == 'all':
            data_folder = os.path.join(args['eeg_root'], self.task)
            if not os.path.exists(data_folder):
                raise FileNotFoundError(f"Subject folder not found: {data_folder}")
            trials, sentences, masks = [], [], []
            for file in os.listdir(data_folder):
                if 'data' in file and split in file:
                    data_path = os.path.join(data_folder, file)
                    label_path = os.path.join(data_folder, f'{file[:-8]}label.npy')
                    mask_path = os.path.join(data_folder, f'{file[:-8]}mask.npy')
                    trials.append(np.load(data_path))
                    sentences.append(np.load(label_path))
                    masks.append(np.load(mask_path))
            trials = np.concatenate(trials, axis=0)
            sentences = np.concatenate(sentences, axis=0)
            masks = np.concatenate(masks, axis=0)
        else:
            if dataset == 'ChineseEEG2':
                data_path = os.path.join(args['eeg_root'], self.task, f"sub-{self.subj}_{split}_data.npy")
                label_path = os.path.join(args['eeg_root'], self.task, f"sub-{self.subj}_{split}_label.npy")
                mask_path = os.path.join(args['eeg_root'], self.task, f"sub-{self.subj}_{split}_mask.npy")
            else:
                data_path = os.path.join(args['eeg_root'], self.task, f"{self.subj}_{split}_data.npy")
                label_path = os.path.join(args['eeg_root'], self.task, f"{self.subj}_{split}_label.npy")
                mask_path = os.path.join(args['eeg_root'], self.task, f"{self.subj}_{split}_mask.npy")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Subject files not found: {data_path}")
            
            trials = np.load(data_path)
            sentences = np.load(label_path)
            masks = np.load(mask_path)
        
        full_data = []
        for i, tr in enumerate(trials):
            sentence = str(sentences[i].item())
            if sentence not in token_lookup:
                continue
            raw_tokens = token_lookup[sentence]
            eeg_ct = tr.astype(np.float32) * 1e6          
            eeg = torch.from_numpy(eeg_ct)
            mask = torch.from_numpy(masks[i].astype(np.float32))        
            mask = 1-mask[0]
            full_data.append((eeg, mask, raw_tokens))

        if not full_data:
            raise ValueError(f"No valid trials found for subject {self.subj}, split={split}")

        self.data = []
        for eeg, mask, toks in full_data:
            cluster_toks = [self.word_cluster[i] for i in toks]
            self.data.append({
                "eeg": eeg,
                "tokens": cluster_toks,
                "mask": mask
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
class PrivateConceptDataset(Dataset):
    def __init__(self, subj, task, args, split="train", dataset="private"):
        assert split in ("train", "val")
        self.split = split        
        self.mask_len = 3000
        with open(args['segmentation_path'], "r", encoding="utf-8") as f:
            raw = json.load(f)
        seg_df = pd.DataFrame(raw)
        with open(args["word2cluster_path"], 'r') as f:
            self.word2cluster = json.load(f)
        
        required_cols_any = ["sentence"]
        missing = [c for c in required_cols_any if c not in seg_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        seg_df = seg_df.dropna(subset=["sentence"])
        seg_df["sentence"] = seg_df["sentence"].astype(str).str.strip()
        token_cols = [f"col{i}" for i in range(1, 11) if f"col{i}" in seg_df.columns]

        total_len = len(seg_df)
        split_idx = int(total_len * 0.8)
        if self.split == "train":
            seg_df = seg_df.iloc[:split_idx].reset_index(drop=True)
        else:
            seg_df = seg_df.iloc[split_idx:].reset_index(drop=True)

        self.data, self.labels, self.length = [], [], []
       
        for _, row in seg_df.iterrows():
            toks = []
            for c in token_cols:
                val = row.get(c, None)
                if pd.notna(val):
                    sval = str(val).strip()
                    if sval and sval.lower() != "nan" and sval in self.word2cluster:
                        toks.append(sval)
            filename = row["new_name"]
            eeg_path = os.path.join(args['eeg_root'], filename)
            if not os.path.exists(eeg_path):
                continue
            eeg = np.load(eeg_path)
            cluster_toks = [self.word2cluster[i] for i in toks]
            self.data.append({
                "eeg": torch.tensor(eeg, dtype=torch.float32),
                "tokens": cluster_toks,
                "mask": torch.ones(self.mask_len)
            })
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    


def load_word_emb(args):
    cluster_emb = torch.load(args['cluster_emb_path'])
    with open(args['word2cluster_path'], 'r') as f:
        word2cluster = json.load(f)
    return cluster_emb, word2cluster

                              
                          
                              
SPECIALS = ["<PAD>", "<BOS>", "<UNK>"]           

def build_vocab_from_dataset(dataset1, dataset2, min_freq=1):
                                                  
    counter = Counter()
    for sample in dataset1:
        counter.update(sample["tokens"])
    for sample in dataset2:
        counter.update(sample["tokens"])
    tokens = [tok for tok, c in counter.items() if c >= min_freq]
    id2token = SPECIALS + tokens
    token2id = {t: i for i, t in enumerate(id2token)}
    return token2id, id2token
