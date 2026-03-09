import argparse

if __name__ == "__main__":
    _parser = argparse.ArgumentParser("Multi-Cls baseline (public release)")
    _parser.add_argument("--config", required=True, help="Path to baseline config JSON.")
    _parser.parse_args()
    raise RuntimeError(
        "This baseline is intentionally disabled in the public release. "
        "Only core BrainMosaic-SID training is runnable."
    )

import os
import math
import json
import torch
import torch.nn as nn
import numpy
import csv
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import get_linear_schedule_with_warmup
from eegcnn import EEGcnn, PositionalEncoding
from collections import defaultdict, Counter
from sklearn.metrics import f1_score

from eeg import EEGConceptDataset, ConceptDataset, PrivateConceptDataset, load_word_emb, custom_collate_fn
from KMA import hungarian_match
from config_args import getPath

class EEGclassification(torch.nn.Module):
    def __init__(self, chans=122, timestamp=165, cls=3, dropout1=0.1, dropout2=0.1, layer=0, pooling=None, size1=8, size2=8, feel1=125, feel2=25):
        super().__init__()
        self.eegcnn = EEGcnn(Chans=chans, kernLength1=feel1, kernLength2=feel2, F1=size1, D=size2, F2=size1*size2, P1=2, P2=5, dropoutRate=dropout1)
        self.linear = torch.nn.Linear(timestamp*size1*size2 if pooling is None else size1*size2, cls)
        self.layer = layer
        self.pooling = pooling
        if self.layer > 0:
            self.poscode = PositionalEncoding(size1*size2, dropout=dropout2, max_len=timestamp)
            self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=size1*size2, nhead=size1*size2//8, dim_feedforward=4*size1*size2, batch_first=True, dropout=dropout2), num_layers=self.layer)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
                                                             
        hidden = self.eegcnn(inputs).permute(0, 2, 1)
        if self.layer > 0:        
            hidden = self.poscode(hidden)
            hidden = self.encoder(hidden, src_key_padding_mask=(mask.bool()==False))
        if self.pooling is None: hidden = torch.flatten(hidden, start_dim=1)
        if self.pooling == "mean": hidden = torch.sum(hidden*mask.unsqueeze(dim=2), dim=1)/torch.sum(mask, dim=1).unsqueeze(dim=1)
        if self.pooling == "sums": hidden = torch.sum(hidden*mask.unsqueeze(dim=2), dim=1)
        if self.pooling == "tops": hidden = hidden[:, 0, :]
        output = self.linear(hidden)
                                                     
        return output


def init_csv_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'matching_acc', 'mean_cosine'])

def append_to_csv(file_path, dataset_name, matching_acc, mean_cosine):
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([dataset_name, matching_acc, mean_cosine])
        

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu_id', type=int, default=3)
parser.add_argument('--is_train', type=bool, default=False)
parser.add_argument('--lr1', type=float, default=5e-4)        
parser.add_argument('--wd1', type=float, default=0.01)
parser.add_argument('--tau', type=float, default=0.0)
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--train_log', type=int, default=10)
parser.add_argument('--evals_log', type=int, default=100)
parser.add_argument('--checkpoint_log', type=int, default=100)
parser.add_argument('--chans', type=int, default=122)
parser.add_argument('--timestamp', type=int, default=600)        
parser.add_argument('--pooling', type=str, default='mean')        
parser.add_argument('--size1', type=int, default=8)
parser.add_argument('--size2', type=int, default=8)
parser.add_argument('--feel1', type=int, default=20)         
parser.add_argument('--feel2', type=int, default=10)        
parser.add_argument('--cls', type=int, default=3074)
parser.add_argument('--layer', type=int, default=1)
parser.add_argument('--dropout1', type=float, default=0.5)       
parser.add_argument('--dropout2', type=float, default=0.5)       
parser.add_argument('--sub', type=str, default='08')
parser.add_argument('--rand_guess', type=int, default=0)                                                                                          
parser.add_argument('--task', type=str, default='PassiveListening')
parser.add_argument('--dataset', type=str, default='ChineseEEG2')
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--match', type=str, default='')
parser.add_argument('--subset_ratio', type=float, default=1.0)
args = parser.parse_args()
args = getPath(args)
print(args)

data_folder = os.path.join(args.eeg_path, args.task, f'{args.sub}_train_data.npy')
if 'Zuco' in args.dataset and (not os.path.exists(data_folder)):
    raise FileNotFoundError(f"Subject folder not found: {data_folder}")

seed = args.seed
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = EEGclassification(chans=args.chans, timestamp=args.timestamp, cls=args.cls, dropout1=args.dropout1, dropout2=args.dropout2, layer=args.layer, pooling=args.pooling, size1=args.size1, size2=args.size2, feel1=args.feel1, feel2=args.feel2)
concept2emb, word2cluster = load_word_emb(args)

if args.dataset == "Chisco":    
    trainset = EEGConceptDataset(args=args, word2cluster=word2cluster, split='train')
    validset = EEGConceptDataset(args=args, word2cluster=word2cluster, split='val')
elif args.dataset == 'private':
    trainset = PrivateConceptDataset(args=args, word2cluster=word2cluster, split='train')
    validset = PrivateConceptDataset(args=args, word2cluster=word2cluster, split='val')
else:
    trainset = ConceptDataset(args=args, word2cluster=word2cluster, split='train')
    validset = ConceptDataset(args=args, word2cluster=word2cluster, split='val')


trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
validloader = DataLoader(validset, batch_size=args.batch, shuffle=True)
label_freqs = torch.tensor(trainset.label_freqs)
print(label_freqs)
print(len(trainset), len(trainloader))
print(len(validset), len(validloader))


def calculate_similarity_metrics(pred_concepts, true_concepts, concept2emb):
    total_pairs, correct_pairs = 0, 0
    cosine_sum = 0.0
    embs = list(concept2emb.values())[0]
    
    for pred_list, true_list in zip(pred_concepts, true_concepts):
        pred_embs = [embs[concept] for concept in pred_list]
        true_embs = [embs[concept] for concept in true_list]
        
        if not pred_embs or not true_embs:
            continue
        pred_tensor = torch.stack(pred_embs)
        true_tensor = torch.stack(true_embs)

        similarity_matrix = torch.nn.functional.cosine_similarity(
            pred_tensor.unsqueeze(1), true_tensor.unsqueeze(0), dim=2
        )
        
        for true_label in true_list:
            if true_label in pred_list:
                correct_pairs += 1
        cosine_sum += torch.sum(similarity_matrix).item()
        total_pairs += similarity_matrix.numel()
    
    matching_acc = correct_pairs / total_pairs if total_pairs else 0
    mean_cosine = cosine_sum / total_pairs if total_pairs > 0 else 0
    
    return matching_acc, mean_cosine


def valid(args, valid_dataloader, device, model):
    model.eval()
    all_pred_concepts = []
    all_true_concepts = []
    
    with torch.no_grad():
        for idy, (valid_input_features, valid_labels, valid_length) in enumerate(valid_dataloader):
            valid_input_features = valid_input_features.to(device)
            valid_length = valid_length.to(device)
            
            outputs = model(valid_input_features, valid_length)
            topk_values, topk_indices = torch.topk(outputs, k=args.topk, dim=1)
            
                                                        
            batch_pred_concepts = []
            for indices in topk_indices:
                pred_concepts = [idx.item() for idx in indices.cpu().numpy()]
                batch_pred_concepts.append(pred_concepts)
            
                    
            batch_true_concepts = []
            for words in valid_labels:
                true_concepts = [word.item() for word in words if word >= 0]
                batch_true_concepts.append(true_concepts)
            
            all_pred_concepts.extend(batch_pred_concepts)
            all_true_concepts.extend(batch_true_concepts)
    
            
    if args.match == 'kma':
        matching_acc, mean_cosine = hungarian_match(
            all_pred_concepts, all_true_concepts, concept2emb
        )
    else:
        matching_acc, mean_cosine = calculate_similarity_metrics(
            all_pred_concepts, all_true_concepts, concept2emb
        )   

    return matching_acc, mean_cosine, all_pred_concepts, all_true_concepts
        

def calculate_loss(outputs, labels, criterion):    
    batch_size = outputs.shape[0]
    total_loss = 0.0
    valid_samples = 0
    
    for i in range(batch_size):
        valid_indices = [l.item() for l in labels[i] if l >= 0]                           
        
        if len(valid_indices) > 0:
            target = torch.zeros_like(outputs[i])
            target[valid_indices] = 1.0
            
            loss = criterion(outputs[i].unsqueeze(0), target.unsqueeze(0))
            total_loss += loss
            valid_samples += 1
    
    return total_loss / valid_samples if valid_samples > 0 else total_loss


def train(train_dataloader, valid_dataloader, model, config, label_frequency):
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() and config.gpu_id < torch.cuda.device_count() else "cpu")
    model.to(device)
    print(device)

    label_frequency = torch.log(label_frequency.pow(config.tau)+1e-12).unsqueeze(dim=0)
    label_frequency = label_frequency.to(device)
    print(label_frequency.dtype, label_frequency.shape, label_frequency)

    training_step = len(train_dataloader)*config.epoch
    warmup_step = math.ceil(training_step*config.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr1, weight_decay=config.wd1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, training_step)
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    best_matching_acc = 0.0  
    best_mean_cosine = 0.0
    for epoch in range(config.epoch):
        for idx, (input_features, labels, length) in enumerate(train_dataloader):
            step = epoch*len(train_dataloader)+idx+1
            model.train()

            input_features = input_features.to(device)
            labels = labels.to(device)
            length = length.to(device)

            optimizer.zero_grad()
            output = model(input_features, length)
            loss = calculate_loss(output, labels, criterion)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            
        print("step:{}(epoch{} {}/{}) loss:{}".format(step, epoch, idx, len(train_dataloader), running_loss/config.train_log))
        running_loss = 0.0
        matching_acc, mean_cosine, _, _ = valid(config, valid_dataloader, device, model)
        print(f"Epoch {epoch}: Matching Acc: {matching_acc:.4f}, Mean Cosine: {mean_cosine:.4f}")
        
                
        if matching_acc > best_matching_acc or (matching_acc == best_matching_acc and mean_cosine > best_mean_cosine):
            best_matching_acc = matching_acc
            best_mean_cosine = mean_cosine
            model_path = os.path.join(config.checkpoint_path, f'checkpoint_best_concept_{config.task}_{config.sub}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with Matching Acc: {matching_acc:.4f}, Mean Cosine: {mean_cosine:.4f}")

    print(f"Final Results - Best Matching Acc: {best_matching_acc:.4f}, Best Mean Cosine: {best_mean_cosine:.4f}")
    return best_matching_acc, best_mean_cosine
    

if not os.path.exists(args.checkpoint_path): os.mkdir(args.checkpoint_path)
if not args.is_train:
    device = f"cuda:{args.gpu_id}"
    label_path = os.path.join(args.checkpoint_path, f'best_label_concept_{args.task}_{args.sub}.json')
    model_path = os.path.join(args.checkpoint_path, f'checkpoint_best_concept_{args.task}_{args.sub}.pt')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(args.checkpoint_path, f'checkpoint_best_concept_{args.sub}.pt')
                                                                                                                                            
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)
    best_matching_acc, best_mean_cosine, pred_concepts, true_concepts = valid(args, validloader, device, model)
    label_dict = {}
    label_dict['ground_truth'] = [', '.join(str(item) for item in t_concept) for t_concept in true_concepts]
    label_dict['predict'] = [', '.join(str(item) for item in p_concept) for p_concept in pred_concepts]
    with open(label_path, 'w') as f:
        json.dump(label_dict, f, indent=4, ensure_ascii=False)
else:
    best_matching_acc, best_mean_cosine = train(trainloader, validloader, model, args, label_freqs)
    
    if not os.path.exists('ans.csv'):
        init_csv_file('ans.csv')
    dataset_name = f"{args.dataset}_{args.sub}_{args.task}"
    append_to_csv('ans.csv', dataset_name, best_matching_acc, best_mean_cosine)
