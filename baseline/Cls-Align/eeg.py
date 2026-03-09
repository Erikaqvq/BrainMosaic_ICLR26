import os
import json
import torch
import pickle
import pdb
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

def adjust_mask(mask, patch_size, mask_threshold=0.8):
    mask = 1-torch.from_numpy(mask).float()
    C, T = mask.shape
    N_patches = T // patch_size
    if N_patches * patch_size != T:
        mask = mask[:, : N_patches * patch_size]
    mask_reshaped = mask.reshape(C, N_patches, patch_size)
    compressed_mask = (mask_reshaped.mean(dim=-1) > mask_threshold)
    
    return compressed_mask.float()[0]                   


def filter_word(args, row, token_cols):
    toks = [str(row[c]).strip() for c in token_cols if pd.notna(row[c]) and str(row[c]).strip()]
    filter_list = FILTER_POS_INITIALS
    if 'Zuco' in args.dataset:
        filter_list = FILTER_POS_INITIALS_EN
    with open(args.filter_path, 'r') as f:
        total_data = json.load(f)
    word2pos = {item['key']: item['pos'] for item in total_data}
    toks = [tok for tok in toks if tok in word2pos and word2pos[tok] not in filter_list]
    return toks
    

def load_word_emb(args):
    cluster_emb = torch.load(args.cluster_emb_path)
    with open(args.word2cluster_path, 'r') as f:
        word2cluster = json.load(f)
    return cluster_emb, word2cluster


def custom_collate_fn(batch):
    data_batch = [item[0] for item in batch]
    labels_batch = torch.tensor([item[1] for item in batch])
    length_batch = torch.stack([item[2] for item in batch])
    eeg_batch = torch.stack([item['eeg'] for item in data_batch])
    
    words_batch = [item['words'] for item in data_batch]
    return {'eeg': eeg_batch, 'words': words_batch}, labels_batch, length_batch


class EEGClsDataset(Dataset):
    def __init__(self, args, split="train"):
        assert split in ("train", "val")
        self.split = split
        self.subj = getattr(args, "sub", "01")
        self.task = getattr(args, "task", "read")
        self.mask_len = args.timestamp

        segmentation_path = args.segmentation_path
        eeg_root = args.eeg_path
        self.dim = getattr(args, "hidden_dim", 128)
        self.freq_threshold = getattr(args, "freq_threshold", 0)

        seg_df = pd.read_excel(segmentation_path)
        seg_df = normalize_segmentation_columns(seg_df)
        seg_df = seg_df.dropna(subset=["sentence"])
        seg_df["sentence"] = seg_df["sentence"].astype(str).str.strip()

        token_cols = [c for c in seg_df.columns if str(c).startswith("token")]
        token_lookup = {}
        for _, row in seg_df.iterrows():
            sent = row["sentence"]
            toks = [str(row[c]).strip() for c in token_cols if pd.notna(row[c]) and str(row[c]).strip()]
            token_lookup[sent] = toks

        valid_run_ids = set([f"0{i}" for i in range(1, 46)])

        with open(args.label_path, 'r') as f:
            label_dict = json.load(f)

        full_data = [] 
        for subj in [self.subj]:
            subj_path = os.path.join(eeg_root, f"sub-{subj}", "eeg")
            if not os.path.exists(subj_path):
                print(f"Subject folder not found: {subj_path}")
                continue

            for fn in sorted(os.listdir(subj_path)):
                if not (fn.endswith(".pkl") and f"sub-{subj}" in fn and f"task-{self.task}" in fn):
                    continue
                try:
                    run_id = fn.split("run-")[1].split("_")[0]
                except IndexError:
                    print(f"cannot parse run id: {fn}")
                    continue
                if run_id not in valid_run_ids:
                    continue

                with open(os.path.join(subj_path, fn), "rb") as f:
                    trials = pickle.load(f)
                length = len(trials)
                train_len = int(length*0.8)
                if split == "train":
                    trials = trials[:train_len]
                else:
                    trials = trials[train_len:]
                for tr in trials:
                    sentence = str(tr.get("text", "")).strip()
                    raw_tokens = token_lookup[sentence]
                    eeg = tr["input_features"][0, :122, :].astype(np.float32) * 1e6
                    full_data.append((eeg, sentence, raw_tokens)) 

        if not full_data:
            raise ValueError(f"No valid trials found for subject {self.subj}, split={split}")

        self.data = []
        self.labels = []
        self.length = []
        for eeg_ct, sentence, raw_tokens in full_data:
            eeg_ct = torch.from_numpy(eeg_ct[:122, :2501])
            label = label_dict[sentence]
            eeg_info = {'eeg': eeg_ct, 'words': raw_tokens}
            self.data.append(eeg_info)
            self.labels.append(label)
            self.length.append(torch.ones(self.mask_len))
                         

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.length[idx]
    
    
class ClsDataset(Dataset):
    def __init__(self, args, split="train"):
        assert split in ("train", "val")
        self.split = split
        self.subj = getattr(args, "sub", "f1")
        self.task = getattr(args, "task", "ReadingAloud")
        segmentation_path = args.segmentation_path
        eeg_root = args.eeg_path
        self.dim = getattr(args, "hidden_dim", 128)
        self.mask_len = args.timestamp
        with open(args.label_path, 'r') as f:
            sen2id = json.load(f)

                                     
        seg_df = pd.read_csv(segmentation_path, sep=',')
        seg_df = normalize_segmentation_columns(seg_df)
        seg_df = seg_df.dropna(subset=["sentence"])
        seg_df["sentence"] = seg_df["sentence"].astype(str).str.strip()

        token_cols = [c for c in seg_df.columns if str(c).startswith("token")]
        token_lookup = {}
        for _, row in seg_df.iterrows():
            sent = row["sentence"]
            toks = filter_word(args, row, token_cols)
            token_lookup[sent] = toks 
        
                    
        data_folder = os.path.join(eeg_root, self.task)
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Subject folder not found: {data_folder}")
        if self.subj == 'all':
            trials, sentences, masks = [], [], []
            for file in os.listdir(data_folder):
                if 'data' in file and split in file:
                    data_path = os.path.join(data_folder, file)
                    label_path = os.path.join(data_folder, f'{file[:-8]}label.npy')
                    mask_path = os.path.join(data_folder, f'{file[:-8]}mask.npy')
                    masks.append(np.load(mask_path)) 
                    trials.append(np.load(data_path))
                    sentences.append(np.load(label_path))
            trials = np.concatenate(trials, axis=0)
            sentences = np.concatenate(sentences, axis=0)
            masks = np.concatenate(masks, axis=0)
        else:     
            if args.dataset == 'ChineseEEG2':
                data_path = os.path.join(data_folder, f"sub-{self.subj}_{split}_data.npy")                    
                label_path = os.path.join(data_folder, f"sub-{self.subj}_{split}_label.npy")
                mask_path = os.path.join(data_folder, f"sub-{self.subj}_{split}_mask.npy")
            else:
                data_path = os.path.join(data_folder, f"{self.subj}_{split}_data.npy")                    
                label_path = os.path.join(data_folder, f"{self.subj}_{split}_label.npy")
                mask_path = os.path.join(data_folder, f"{self.subj}_{split}_mask.npy")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Subject files not found: {data_path}")
            
            trials = np.load(data_path)
            sentences = np.load(label_path)
            masks = np.load(mask_path)
        
        full_data = []
        for i, tr in enumerate(trials):
            sentence = str(sentences[i].item())
            raw_tokens = token_lookup[sentence]
            eeg_ct = tr[:args.chans, :].astype(np.float32) * 1e6          
            eeg = torch.from_numpy(eeg_ct)
            mask = adjust_mask(mask=masks[i], patch_size=10)
            full_data.append((eeg, mask, sentence, raw_tokens))

        if not full_data:
            raise ValueError(f"No valid trials found for subject {self.subj}, split={split}")

        self.data = []
        self.labels = []
        self.length = []
        for eeg, mask, sentence, raw_tokens in full_data:
            sen_id = sen2id[sentence]
            eeg_info = {'eeg': eeg, 'words': raw_tokens}
            self.data.append(eeg_info)
            self.labels.append(sen_id)
            self.length.append(mask)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.length[idx]
    

class EEGConceptDataset(Dataset):
    def __init__(self, args, word2cluster, split="train"):
        assert split in ("train", "val")
        self.split = split
        self.subj = getattr(args, "sub", "01")
        self.task = getattr(args, "task", "read")
        self.mask_len = args.timestamp
        concept_counts = torch.zeros(args.cls)

        segmentation_path = args.segmentation_path
        eeg_root = args.eeg_path
        self.word2cluster = word2cluster
        self.dim = getattr(args, "hidden_dim", 128)

        seg_df = pd.read_excel(segmentation_path)
        seg_df = normalize_segmentation_columns(seg_df)
        seg_df = seg_df.dropna(subset=["sentence"])
        seg_df["sentence"] = seg_df["sentence"].astype(str).str.strip()

        token_cols = [c for c in seg_df.columns if str(c).startswith("token")]
        token_lookup = {}
        for _, row in seg_df.iterrows():
            sent = row["sentence"]
            toks = filter_word(args, row, token_cols)
            token_lookup[sent] = toks
            for concept in toks:
                if concept in self.word2cluster:
                    idx = self.word2cluster[concept]
                    concept_counts[idx] += 1

        valid_run_ids = set([f"0{i}" for i in range(1, 46)])
        
        full_data = [] 
        for subj in [self.subj]:
            subj_path = os.path.join(eeg_root, f"sub-{subj}", "eeg")
            if not os.path.exists(subj_path):
                print(f"Subject folder not found: {subj_path}")
                continue

            for fn in sorted(os.listdir(subj_path)):
                if not (fn.endswith(".pkl") and f"sub-{subj}" in fn and f"task-{self.task}" in fn):
                    continue
                try:
                    run_id = fn.split("run-")[1].split("_")[0]
                except IndexError:
                    print(f"cannot parse run id: {fn}")
                    continue
                if run_id not in valid_run_ids:
                    continue

                with open(os.path.join(subj_path, fn), "rb") as f:
                    trials = pickle.load(f)
                length = len(trials)
                train_len = int(length*0.8)
                if split == "train":
                    trials = trials[:train_len]
                else:
                    trials = trials[train_len:]
                for tr in trials:
                    sentence = str(tr.get("text", "")).strip()
                    raw_tokens = token_lookup[sentence]
                    eeg = tr["input_features"][0, :122, :].astype(np.float32) * 1e6
                    full_data.append((eeg, sentence, raw_tokens)) 

        if not full_data:
            raise ValueError(f"No valid trials found for subject {self.subj}, split={split}")

        self.label_freqs = concept_counts / len(full_data)
        self.data = []
        self.labels = []
        self.length = []
        for eeg_ct, sentence, raw_tokens in full_data:
            eeg_ct = torch.from_numpy(eeg_ct[:122, :])
            label_emb = [self.word2cluster[i] for i in raw_tokens][:args.topk]
            label_emb += [-100]*(args.topk-len(label_emb))
            label_emb = torch.from_numpy(np.array(label_emb))
            self.data.append(eeg_ct)
            self.labels.append(label_emb)
            self.length.append(torch.ones(self.mask_len))
                         

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.length[idx]
    
    
class ConceptDataset(Dataset):
    def __init__(self, args, word2cluster, split="train"):
        assert split in ("train", "val")
        self.split = split
        self.subj = getattr(args, "sub", "f1")
        self.task = getattr(args, "task", "ReadingAloud")
        segmentation_path = args.segmentation_path
        eeg_root = args.eeg_path
        self.dim = getattr(args, "hidden_dim", 128)
        self.mask_len = args.timestamp
        concept_counts = torch.zeros(args.cls)
        self.word2cluster = word2cluster

                                     
        encodings = ['utf-8', 'gbk']
        for encoding in encodings:
            try:
                seg_df = pd.read_csv(segmentation_path, sep=',', encoding=encoding)
                print(f"Loaded file with {encoding} encoding")
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
            toks = filter_word(args, row, token_cols)
            token_lookup[sent] = toks 
        
                    
        if self.subj == 'all':
            data_folder = os.path.join(eeg_root, self.task)
            if not os.path.exists(data_folder):
                raise FileNotFoundError(f"Subject folder not found: {data_folder}")
            trials, sentences, masks = [], [], []
            for file in os.listdir(data_folder):
                if 'data' in file and split in file:
                    data_path = os.path.join(data_folder, file)
                    label_path = os.path.join(data_folder, f'{file[:-8]}label.npy')
                    mask_path = os.path.join(data_folder, f'{file[:-8]}mask.npy')
                    masks.append(np.load(mask_path)) 
                    trials.append(np.load(data_path))
                    sentences.append(np.load(label_path))
            trials = np.concatenate(trials, axis=0)
            sentences = np.concatenate(sentences, axis=0)
            masks = np.concatenate(masks, axis=0)
        else:
            if args.dataset == 'ChineseEEG2':
                data_path = os.path.join(eeg_root, self.task, f"sub-{self.subj}_{split}_data.npy")                    
                label_path = os.path.join(eeg_root, self.task, f"sub-{self.subj}_{split}_label.npy")
                mask_path = os.path.join(eeg_root, self.task, f"sub-{self.subj}_{split}_mask.npy")
            else:
                data_path = os.path.join(eeg_root, self.task, f"{self.subj}_{split}_data.npy")                    
                label_path = os.path.join(eeg_root, self.task, f"{self.subj}_{split}_label.npy")
                if args.task == 'task2-NR':
                    label_path = os.path.join(eeg_root, self.task, f"{self.subj}_{split}_label_cleaned.npy")
                mask_path = os.path.join(eeg_root, self.task, f"{self.subj}_{split}_mask.npy")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Subject files not found: {data_path}")
            
            trials = np.load(data_path)
            sentences = np.load(label_path)
            masks = np.load(mask_path)
        
        full_data = []
        for i, tr in enumerate(trials):
            sentence = str(sentences[i].item())
            raw_tokens = token_lookup[sentence]
            for concept in toks:
                if concept in self.word2cluster:
                    idx = self.word2cluster[concept]
                    concept_counts[idx] += 1
            eeg_ct = tr[:args.chans, :].astype(np.float32) * 1e6          
            eeg = torch.from_numpy(eeg_ct)
            mask = adjust_mask(mask=masks[i], patch_size=10)
            full_data.append((eeg, mask, sentence, raw_tokens))

        if not full_data:
            raise ValueError(f"No valid trials found for subject {self.subj}, split={split}")
        self.label_freqs = concept_counts / len(full_data)

        self.data = []
        self.labels = []
        self.length = []
        for eeg, mask, sentence, raw_tokens in full_data:
            label_emb = [self.word2cluster[i] for i in raw_tokens if i in self.word2cluster][:args.topk]
            label_emb += [-100]*(args.topk-len(label_emb))
            label_emb = torch.from_numpy(np.array(label_emb))
            self.data.append(eeg)
            self.labels.append(label_emb)
            self.length.append(mask)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.length[idx]
    
    
class EEGSentenceDataset(Dataset):
    def __init__(self, args, split="train"):
        assert split in ("train", "val")
        self.split = split
        self.subj = getattr(args, "sub", "01")
        self.task = getattr(args, "task", "read")
        eeg_root = args.eeg_path
        self.dim = getattr(args, "hidden_dim", 128)
        self.mask_len = args.timestamp

        valid_run_ids = set([f"0{i}" for i in range(1, 46)])
        sen_emb = torch.load(args.sen_emb_path)
        self.total_sentences = sen_emb['sentences']
        self.sen_embs = sen_emb['embeddings']
        
        self.data = []
        self.labels = []
        self.length = [] 
        self.target_embs = []
        for subj in [self.subj]:
            subj_path = os.path.join(eeg_root, f"sub-{subj}", "eeg")
            if not os.path.exists(subj_path):
                print(f"Subject folder not found: {subj_path}")
                continue

            for fn in sorted(os.listdir(subj_path)):
                if not (fn.endswith(".pkl") and f"sub-{subj}" in fn and f"task-{self.task}" in fn):
                    continue
                try:
                    run_id = fn.split("run-")[1].split("_")[0]
                except IndexError:
                    print(f"cannot parse run id: {fn}")
                    continue
                if run_id not in valid_run_ids:
                    continue

                with open(os.path.join(subj_path, fn), "rb") as f:
                    trials = pickle.load(f)
                length = len(trials)
                train_len = int(length*0.8)
                if split == "train":
                    trials = trials[:train_len]
                else:
                    trials = trials[train_len:]
                for tr in trials:
                    sentence = str(tr.get("text", "")).strip()
                    index = self.total_sentences.index(sentence)
                    emb = self.sen_embs[index]
                    eeg = tr["input_features"][0, :122, :].astype(np.float32) * 1e6
                    self.data.append(eeg)
                    self.labels.append(sentence)
                    self.length.append(torch.ones(self.mask_len))
                    self.target_embs.append(emb[:args.emb_dim])           

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.length[idx], self.target_embs[idx]
    
    
class ConceptSenDataset(Dataset):
    def __init__(self, args, split="train"):
        assert split in ("train", "val")
        self.split = split
        self.subj = getattr(args, "sub", "f1")
        self.task = getattr(args, "task", "ReadingAloud")
        eeg_root = args.eeg_path
        self.dim = getattr(args, "hidden_dim", 128)
        self.mask_len = args.timestamp
        sen_emb = torch.load(args.sen_emb_path)
        self.total_sentences = sen_emb['sentences']
        self.sen_embs = sen_emb['embeddings']

                    
        if self.subj == 'all':
            data_folder = os.path.join(eeg_root, self.task)
            if not os.path.exists(data_folder):
                raise FileNotFoundError(f"Subject folder not found: {data_folder}")
            trials, sentences, masks = [], [], []
            for file in os.listdir(data_folder):
                if 'data' in file and split in file:
                    data_path = os.path.join(data_folder, file)
                    label_path = os.path.join(data_folder, f'{file[:-8]}label.npy')
                    mask_path = os.path.join(data_folder, f'{file[:-8]}mask.npy')
                    masks.append(np.load(mask_path)) 
                    trials.append(np.load(data_path))
                    sentences.append(np.load(label_path))
            trials = np.concatenate(trials, axis=0)
            sentences = np.concatenate(sentences, axis=0)
            masks = np.concatenate(masks, axis=0)
        else:
            if args.dataset == 'ChineseEEG2':
                data_path = os.path.join(eeg_root, self.task, f"sub-{self.subj}_{split}_data.npy")                    
                label_path = os.path.join(eeg_root, self.task, f"sub-{self.subj}_{split}_label.npy")
                mask_path = os.path.join(eeg_root, self.task, f"sub-{self.subj}_{split}_mask.npy")
            else:
                data_path = os.path.join(eeg_root, self.task, f"{self.subj}_{split}_data.npy")                    
                label_path = os.path.join(eeg_root, self.task, f"{self.subj}_{split}_label_cleaned.npy")
                if not os.path.exists(label_path):
                    label_path = os.path.join(eeg_root, self.task, f"{self.subj}_{split}_label.npy")
                mask_path = os.path.join(eeg_root, self.task, f"{self.subj}_{split}_mask.npy")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Subject files not found: {data_path}")
            
            trials = np.load(data_path)
            sentences = np.load(label_path)
            masks = np.load(mask_path)
        
        self.data = []
        self.labels = []
        self.length = []
        self.target_emb = []
        for i, tr in enumerate(trials):
            sentence = str(sentences[i].item())
            if sentence not in self.total_sentences:
                continue
            index = self.total_sentences.index(sentence)
            emb = self.sen_embs[index]
            eeg_ct = tr[:args.chans, :].astype(np.float32) * 1e6          
            eeg = torch.from_numpy(eeg_ct)
            mask = adjust_mask(mask=masks[i], patch_size=10)
            self.data.append(eeg)
            self.labels.append(sentence)
            self.length.append(mask)
            self.target_emb.append(emb[:args.emb_dim])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.length[idx], self.target_emb[idx]


class PrivateClsDataset(Dataset):
    def __init__(self, args, split="train"):
        assert split in ("train", "val")
        self.split = split
        self.mask_len = args.timestamp
        eeg_root = args.eeg_path
        self.dim = getattr(args, "hidden_dim", 128)
        self.freq_threshold = getattr(args, "freq_threshold", 0)
        
        with open(args.segmentation_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        seg_df = pd.DataFrame(raw)

        required_cols_any = ["sentence"]
        missing = [c for c in required_cols_any if c not in seg_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        seg_df = seg_df.dropna(subset=["sentence"])
        seg_df["sentence"] = seg_df["sentence"].astype(str).str.strip()

        total_len = len(seg_df)
        split_idx = int(total_len * 0.8)
        if self.split == "train":
            seg_df = seg_df.iloc[:split_idx].reset_index(drop=True)
        else:
            seg_df = seg_df.iloc[split_idx:].reset_index(drop=True)

        token_cols = [f"col{i}" for i in range(1, 11) if f"col{i}" in seg_df.columns]
        self.data, self.labels, self.length = [], [], []
        token_lookup = {}
        
        for _, row in seg_df.iterrows():
            sent = str(row["sentence"]).strip()

            toks = []
            for c in token_cols:
                val = row.get(c, None)
                if pd.notna(val):
                    sval = str(val).strip()
                    if sval and sval.lower() != "nan":
                        toks.append(sval)

            token_lookup[sent] = toks

            filename = row["new_name"]
            eeg_path = os.path.join(eeg_root, filename)
            if not os.path.exists(eeg_path):
                continue
            eeg = np.load(eeg_path)
            eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
            self.data.append({'eeg':eeg_tensor, 'words': toks})
            self.labels.append(int(row['sentence_concept_id'])-1)
            self.length.append(torch.ones(self.mask_len))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.length[idx]


class PrivateSenDataset(Dataset):
    def __init__(self, args, split="train"):
        assert split in ("train", "val")
        self.split = split
        self.mask_len = args.timestamp
        eeg_root = args.eeg_path
        self.dim = getattr(args, "hidden_dim", 128)
        self.freq_threshold = getattr(args, "freq_threshold", 0)

        sen_emb = torch.load(args.sen_emb_path)
        self.total_sentences = sen_emb['sentences']
        self.sen_embs = sen_emb['embeddings']

        with open(args.segmentation_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        seg_df = pd.DataFrame(raw)

        required_cols_any = ["sentence"]
        missing = [c for c in required_cols_any if c not in seg_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        seg_df = seg_df.dropna(subset=["sentence"])
        seg_df["sentence"] = seg_df["sentence"].astype(str).str.strip()

        total_len = len(seg_df)
        split_idx = int(total_len * 0.8)
        if self.split == "train":
            seg_df = seg_df.iloc[:split_idx].reset_index(drop=True)
        else:
            seg_df = seg_df.iloc[split_idx:].reset_index(drop=True)

        self.data, self.labels, self.length, self.target_emb = [], [], [], []
       
        for _, row in seg_df.iterrows():
            sent = str(row["sentence"]).strip()
            filename = row["new_name"]
            eeg_path = os.path.join(eeg_root, filename)
            if not os.path.exists(eeg_path):
                continue
            eeg = np.load(eeg_path)
            eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
            self.data.append(eeg_tensor)
            self.labels.append(sent)
            self.length.append(torch.ones(self.mask_len))
            index = self.total_sentences.index(sent)
            self.target_emb.append(self.sen_embs[index][:args.emb_dim])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.length[idx], self.target_emb[idx]


class PrivateConceptDataset(Dataset):
    def __init__(self, args, word2cluster, split="train"):
        assert split in ("train", "val")
        self.split = split
        self.mask_len = args.timestamp
        eeg_root = args.eeg_path
        self.dim = getattr(args, "hidden_dim", 128)
        self.freq_threshold = getattr(args, "freq_threshold", 0)
        concept_counts = torch.zeros(args.cls)
        
        with open(args.segmentation_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        seg_df = pd.DataFrame(raw)
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
                    if sval and sval.lower() != "nan" and sval in word2cluster:
                        toks.append(sval)
                        idx = word2cluster[sval]
                        concept_counts[idx] += 1
            
            label_emb = [word2cluster[i] for i in toks][:args.topk]
            label_emb += [-100]*(args.topk-len(label_emb))
            label_emb = torch.from_numpy(np.array(label_emb))
            filename = row["new_name"]
            eeg_path = os.path.join(eeg_root, filename)
            if not os.path.exists(eeg_path):
                continue
            eeg = np.load(eeg_path)
            eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
            self.data.append(eeg_tensor)
            self.labels.append(label_emb)
            self.length.append(torch.ones(self.mask_len))
        self.label_freqs = concept_counts / len(self.data)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.length[idx]
    

def load_dataset(args):
    if args.train_phase == 'clip':
        if args.dataset == "Chisco":
            trainset = EEGSentenceDataset(args=args, split='train')
            validset = EEGSentenceDataset(args=args, split='val')
        elif args.dataset == "private":
            trainset = PrivateSenDataset(args=args, split='train')
            validset = PrivateSenDataset(args=args, split='val')
        else:
            trainset = ConceptSenDataset(args=args, split='train')
            validset = ConceptSenDataset(args=args, split='val')
    else:
        if args.dataset == "Chisco":
            trainset = EEGClsDataset(args=args, split='train')
            validset = EEGClsDataset(args=args, split='val')
        elif args.dataset == "private":
            trainset = PrivateClsDataset(args=args, split='train')
            validset = PrivateClsDataset(args=args, split='val')
        else:
            trainset = ClsDataset(args=args, split='train')
            validset = ClsDataset(args=args, split='val')
    
    if args.train_phase == 'train':
        trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, collate_fn=custom_collate_fn)
        validloader = DataLoader(validset, batch_size=args.batch, shuffle=True, collate_fn=custom_collate_fn)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
        validloader = DataLoader(validset, batch_size=args.batch, shuffle=True)
        
    return trainset, validset, trainloader, validloader
