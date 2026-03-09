import argparse

if __name__ == "__main__":
    _parser = argparse.ArgumentParser("neuro2semantic baseline (public release)")
    _parser.add_argument("--config", required=True, help="Path to baseline config JSON.")
    _parser.parse_args()
    raise RuntimeError(
        "This baseline is intentionally disabled in the public release. "
        "Only core BrainMosaic-SID training is runnable."
    )

import os
import csv
import json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
              
from data_loader import create_dataloader, NeuralDataset
from models import NeuralEncoder
import torch.optim as optim
import random
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from reconstruct_text import reconstruct_text, custom_invert_embeddings
from utils import calculate_bert_score, calculate_bleu_score, load_vec2text_model
import vec2text
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm
from config_args import updateArgs


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): Seed value to set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

def calculate_cosine_similarity(neural_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> float:
    """
    Calculate the average cosine similarity between neural and text embeddings.

    Args:
        neural_embeddings (torch.Tensor): Neural embeddings.
        text_embeddings (torch.Tensor): Text embeddings.

    Returns:
        float: Average cosine similarity.
    """
    return torch.nn.functional.cosine_similarity(neural_embeddings, text_embeddings, dim=-1).mean().item()



def calculate_mse(neural_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> float:
    """
    Calculate the Mean Squared Error (MSE) between neural and text embeddings.

    Args:
        neural_embeddings (torch.Tensor): Neural embeddings.
        text_embeddings (torch.Tensor): Text embeddings.

    Returns:
        float: MSE value.
    """
    mse = torch.nn.functional.mse_loss(neural_embeddings, text_embeddings)
    return mse.item()

def sequence_loss_fn(reconstructed_texts: List[str], original_texts: List[str], corrector: vec2text.trainers.Corrector) -> torch.Tensor:
    """
    Compute the MSE loss between reconstructed and original texts.

    Args:
        reconstructed_texts (List[str]): List of reconstructed text strings.
        original_texts (List[str]): List of original text strings.
        corrector (vec2text.trainers.Corrector): Vec2Text corrector model.

    Returns:
        torch.Tensor: MSE loss value.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
                        
    tokenized_targets = [
        corrector.tokenizer.encode(text, return_tensors='pt').squeeze(0).to(device) 
        for text in original_texts
    ]
    tokenized_outputs = [
        corrector.tokenizer.encode(recon, return_tensors='pt').squeeze(0).to(device) 
        for recon in reconstructed_texts
    ]

                                                      
    padded_targets = pad_sequence(tokenized_targets, batch_first=True, padding_value=corrector.tokenizer.pad_token_id)
    padded_outputs = pad_sequence(tokenized_outputs, batch_first=True, padding_value=corrector.tokenizer.pad_token_id)

                                                    
    min_len = min(padded_outputs.shape[1], padded_targets.shape[1])
    padded_outputs = padded_outputs[:, :min_len].float()                            
    padded_targets = padded_targets[:, :min_len].float()                            

                      
    return F.mse_loss(padded_outputs, padded_targets)





class CLIPLoss(nn.Module):
    """
    CLIP Loss module combining cross-entropy losses for neural-to-text and text-to-neural mappings.
    """
    def __init__(self, temperature: float = 0.07):
        """
        Initialize the CLIPLoss module.

        Args:
            temperature (float, optional): Temperature scaling factor. Defaults to 0.07.
        """
        super(CLIPLoss, self).__init__()
        self.temperature = temperature

    def forward(self, neural_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the CLIP loss.

        Args:
            neural_embeddings (torch.Tensor): Neural embeddings.
            text_embeddings (torch.Tensor): Text embeddings.

        Returns:
            torch.Tensor: Computed CLIP loss.
        """
                                  
        neural_embeddings = F.normalize(neural_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

                                   
        logits = (text_embeddings @ neural_embeddings.T) / self.temperature

                               
        targets = torch.arange(logits.size(0)).long().to(neural_embeddings.device)

                                                        
        loss_neural_to_text = F.cross_entropy(logits, targets)
        loss_text_to_neural = F.cross_entropy(logits.T, targets)

                            
        loss = (loss_neural_to_text + loss_text_to_neural) / 2.0
        return loss
    
    
class CombinedCLIPTripletLoss(nn.Module):
    """
    Combined loss function that integrates CLIP Loss and Triplet Margin Loss.
    """
    def __init__(self, temperature: float = 0.07, margin: float = 1.0, alpha: float = 0.5):
        """
        Initialize the CombinedCLIPTripletLoss module.

        Args:
            temperature (float, optional): Temperature parameter for CLIP Loss. Defaults to 0.07.
            margin (float, optional): Margin for Triplet Margin Loss. Defaults to 1.0.
            alpha (float, optional): Weighting factor between CLIP Loss and Triplet Margin Loss. Defaults to 0.5.
        """
        super(CombinedCLIPTripletLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, neural_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the combined CLIP and Triplet loss.

        Args:
            neural_embeddings (torch.Tensor): Neural embeddings.
            text_embeddings (torch.Tensor): Text embeddings.

        Returns:
            torch.Tensor: Combined loss value.
        """
                                  
        neural_embeddings = F.normalize(neural_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

                           
        logits = (text_embeddings @ neural_embeddings.T) / self.temperature
        targets = torch.arange(logits.size(0)).long().to(neural_embeddings.device)
        loss_neural_to_text = self.cross_entropy_loss(logits, targets)
        loss_text_to_neural = self.cross_entropy_loss(logits.T, targets)
        clip_loss = (loss_neural_to_text + loss_text_to_neural) / 2.0

                                     
        anchor_embeddings = neural_embeddings
        positive_embeddings = text_embeddings
        negative_embeddings = text_embeddings[torch.randperm(text_embeddings.size(0))]

        triplet_margin_loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

                            
        combined_loss = self.alpha * clip_loss + (1 - self.alpha) * triplet_margin_loss
        return combined_loss


def cross_entropy_loss(preds: torch.Tensor, targets: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
    """
    Compute the cross-entropy loss.

    Args:
        preds (torch.Tensor): Predictions.
        targets (torch.Tensor): Targets.
        reduction (str, optional): Reduction method. Defaults to 'none'.

    Returns:
        torch.Tensor: Computed loss.
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def similarity_loss(reconstructed_texts: List[str], original_texts: List[str], model: SentenceTransformer) -> float:
    """
    Calculate the similarity loss between reconstructed and original texts.

    Args:
        reconstructed_texts (List[str]): List of reconstructed text strings.
        original_texts (List[str]): List of original text strings.
        model (SentenceTransformer): Pre-trained sentence transformer model.

    Returns:
        float: Similarity loss value.
    """
    embeddings_recon = model.encode(reconstructed_texts, convert_to_tensor=True)
    embeddings_orig = model.encode(original_texts, convert_to_tensor=True)
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings_recon, embeddings_orig)
    return 1 - cosine_sim.mean().item()


                                                    
def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    corrector: vec2text.trainers.Corrector
) -> tuple:
    """
    Train the neural encoder model for one epoch.

    Args:
        model (nn.Module): Neural encoder model.
        dataloader (DataLoader): Training DataLoader.
        optimizer (optim.Optimizer): Optimizer.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to train on.
        epoch (int): Current epoch number.
        corrector (vec2text.trainers.Corrector): Vec2Text corrector model.

    Returns:
        tuple: Average training loss and average corrector loss.
    """
    model.train()
    corrector.model.train()                                            
    total_loss = 0.0
    total_corrector_loss = 0.0

    for neural_segment, text_embedding, masks, original_texts in dataloader:
        neural_segment = neural_segment.to(device)
        text_embedding = text_embedding.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

                                                 
        neural_features = model(neural_segment, masks)

                               
        loss = loss_fn(neural_features, text_embedding)

                                 
        input_ids = corrector.tokenizer(
            list(original_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).input_ids.to(device)

        input_ids = input_ids.long()
        attention_mask = (input_ids != corrector.tokenizer.pad_token_id).long().to(device)

                                          
        corrector_inputs = {
            'frozen_embeddings': neural_features.float(),
            'input_ids': input_ids,
            'labels': input_ids.long()
        }

                                
        corrector_loss = corrector.compute_loss(corrector.model, corrector_inputs)

                        
        combined_loss = loss + corrector_loss
        combined_loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_corrector_loss += corrector_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_corrector_loss = total_corrector_loss / len(dataloader)

                 
                                       
                                                     
                        
        

    print(f"Epoch {epoch}: Avg Train Loss: {avg_loss:.4f}, Avg Corrector Loss: {avg_corrector_loss:.4f}")

    return avg_loss, avg_corrector_loss


def train_neural_encoder(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    embedding_dir: str
) -> float:
    """
    Train the neural encoder model for one epoch.

    Args:
        model (nn.Module): Neural encoder model.
        dataloader (DataLoader): Training DataLoader.
        optimizer (optim.Optimizer): Optimizer.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to train on.
        epoch (int): Current epoch number.
        embedding_dir (str): Directory to save embeddings.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0.0
    all_neural_embeddings = []
    all_text_embeddings = []

    for neural_segment, text_embedding, masks, original_texts in tqdm(dataloader, disable=False, desc=f'Epoch{epoch}'):
        neural_segment = neural_segment.to(device)
        text_embedding = text_embedding.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

                                                 
        neural_features = model(neural_segment, masks)

                               
        loss = loss_fn(neural_features, text_embedding)
        loss.backward()

                                 
        optimizer.step()

        total_loss += loss.item()
        all_neural_embeddings.append(neural_features.cpu().detach().numpy())
        all_text_embeddings.append(text_embedding.cpu().detach().numpy())

    avg_loss = total_loss / len(dataloader)

                 
                                       
                        
        

    print(f"Epoch {epoch}: Avg Train Loss: {avg_loss:.4f}")

                                  
                         
                                                                                                                                      
                                                                                                                

    return avg_loss


def fine_tune_corrector(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    corrector: vec2text.trainers.Corrector
) -> float:
    """
    Fine-tune the Vec2Text corrector model.

    Args:
        model (nn.Module): Frozen neural encoder model.
        dataloader (DataLoader): Training DataLoader.
        optimizer (optim.Optimizer): Optimizer for the corrector.
        device (torch.device): Device to train on.
        epoch (int): Current epoch number.
        corrector (vec2text.trainers.Corrector): Vec2Text corrector model.

    Returns:
        float: Average corrector loss.
    """
    model.eval()                                                          
    corrector.model.train()                                            
    total_corrector_loss = 0.0

    for neural_segment, text_embedding, masks, original_texts in tqdm(dataloader, disable=False, desc=f'Epoch{epoch}'):
        neural_segment = neural_segment.to(device)
        text_embedding = text_embedding.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
                                                            
            neural_features = model(neural_segment, masks)

                                 
        input_ids = corrector.tokenizer(
            list(original_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).input_ids.to(device)

        input_ids = input_ids.long()
        attention_mask = (input_ids != corrector.tokenizer.pad_token_id).long().to(device)

                                          
        corrector_inputs = {
            'frozen_embeddings': neural_features.float(),
            'input_ids': input_ids,
            'labels': input_ids.long()
        }

                                
        corrector_loss = corrector.compute_loss(corrector.model, corrector_inputs)

                                                       
        corrector_loss.backward()
        optimizer.step()

        total_corrector_loss += corrector_loss.item()

    avg_corrector_loss = total_corrector_loss / len(dataloader)

                 
                                                     
                        
        

    print(f"Epoch {epoch}: Avg Corrector Loss: {avg_corrector_loss:.4f}")

    return avg_corrector_loss


def evaluate(
    args,
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    corrector: vec2text.trainers.Corrector,
    similarity_model: SentenceTransformer,
    n_steps: int,
    save_embeddings: bool = False,
    save_freq: int = 10,
    output_dir: str = "embeddings"
) -> tuple:
    """
    Evaluate the neural encoder model on the validation set.

    Args:
        model (nn.Module): Neural encoder model.
        dataloader (DataLoader): Validation DataLoader.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.
        epoch (int): Current epoch number.
        corrector (vec2text.trainers.Corrector): Vec2Text corrector model.
        similarity_model (SentenceTransformer): Sentence transformer for similarity calculations.
        n_steps (int): Number of steps for reconstruction.
        save_embeddings (bool, optional): Whether to save embeddings. Defaults to False.
        save_freq (int, optional): Frequency to save embeddings. Defaults to 10.
        output_dir (str, optional): Directory to save embeddings. Defaults to "embeddings".

    Returns:
        tuple: Average loss, average cosine similarity, neural embeddings, text embeddings, average BERT score.
    """
    model.eval()
    corrector.model.eval()
    total_loss = 0.0
    total_similarity_loss = 0.0
    total_cosine_similarity = 0.0
    total_mse_loss = 0.0
    total_corrector_loss = 0.0
    total_bleu_score = 0.0
    total_bert_score = 0.0
    max_bert_score = -float('inf')
    best_reconstructed_text = None
    best_original_text = None

    all_neural_embeddings = []
    all_text_embeddings = []
    original_sens, pred_sens = [], []

    with torch.no_grad():
        for neural_segment, text_embedding, masks, original_texts in dataloader:
            neural_segment = neural_segment.to(device)
            text_embedding = text_embedding.to(device)
            masks = masks.to(device)

                                                     
            neural_features = model(neural_segment, masks)

                                   
            loss = loss_fn(neural_features, text_embedding)
            total_loss += loss.item()

                                     
            input_ids = corrector.tokenizer(
                list(original_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).input_ids.to(device)

            input_ids = input_ids.long()
            attention_mask = (input_ids != corrector.tokenizer.pad_token_id).long().to(device)

                                              
            corrector_inputs = {
                'frozen_embeddings': neural_features.float(),
                'input_ids': input_ids,
                'labels': input_ids.long()
            }

                                    
            corrector_loss = corrector.compute_loss(corrector.model, corrector_inputs)
            total_corrector_loss += corrector_loss.item()

                                                          
            reconstructed_texts = reconstruct_text(neural_features, corrector, n_steps=n_steps, target_lang=args.target_lang)
            original_sens += list(original_texts)
            pred_sens += reconstructed_texts
            
                                       
            sim_loss = similarity_loss(reconstructed_texts, original_texts, similarity_model)
            total_similarity_loss += sim_loss

                                        
            cosine_similarity = calculate_cosine_similarity(neural_features, text_embedding)
            mse = calculate_mse(neural_features, text_embedding)

            total_cosine_similarity += cosine_similarity
            total_mse_loss += mse

            bleu_score = calculate_bleu_score(reconstructed_texts, original_texts)
            bert_scores = calculate_bert_score(reconstructed_texts, original_texts)
            total_bleu_score += bleu_score
            total_bert_score += bert_scores[2]  

                                       
            if bert_scores[2] > max_bert_score:
                max_bert_score = bert_scores[2]
                best_reconstructed_text = reconstructed_texts
                best_original_text = original_texts

                                        
            if save_embeddings:
                all_neural_embeddings.append(neural_features.cpu().numpy())
                all_text_embeddings.append(text_embedding.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_similarity_loss = total_similarity_loss / len(dataloader)
    avg_cosine_similarity = total_cosine_similarity / len(dataloader)
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_bleu_score = total_bleu_score / len(dataloader)
    avg_bert_score = total_bert_score / len(dataloader)

    print(f"Epoch {epoch}: Avg Eval Loss: {avg_loss:.4f}, Avg Similarity Loss: {avg_similarity_loss:.4f}, "
          f"Cosine Similarity: {avg_cosine_similarity:.4f}, BLEU: {avg_bleu_score:.4f}, BERT: {avg_bert_score:.4f}")

    if best_reconstructed_text and best_original_text:
        print(f"Best Reconstructed Text with BERT Score {max_bert_score:.4f}: {best_reconstructed_text}")
        print(f"Original Text: {best_original_text}")

                 
                                
                                                                        
                                                      
                                                          
                                   
                                            
                                            
                        
        

    return avg_loss, avg_cosine_similarity, all_neural_embeddings, all_text_embeddings, avg_bert_score, original_sens, pred_sens


def main(args):
    """
    Main function to train the neural encoder model with integrated Vec2Text corrector.

    Args:
        args: Parsed command-line arguments.
    """
                      
    set_seed(42)
    args = updateArgs(args)
    print("Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
                                                                                               
                                   
    
                                               
                                                                                                                 
    
                
                                                                                          
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    similarity_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    similarity_model = similarity_model.to(device)
    
               
                                           
                                                                  
                             
                                     
                               
                               
                                   
                           
                                                             
                                                    
           
           
                                                    
                             
                                     
                               
                               
                                   
                           
                                                            
           

                                                        
                                                
                                            
                                              
                                                                                               

                                    
                            
                                         
                           
                                              
           
                                  
                          
                                         
                            
                                              
           
    
    train_loader, val_loader, input_dim = create_dataloader(args)
    
                                                    
    if args.embedding_model_name == "gtr-t5-base":
        embedding_dim = 768
        vec2text_model = 'gtr-base'
    else:
        embedding_dim = 1536
        vec2text_model = 'text-embedding-ada-002'

    model = NeuralEncoder(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    loss_fn = CombinedCLIPTripletLoss(temperature=args.temperature, margin=args.margin, alpha=args.alpha).to(device)

                             
    corrector = load_vec2text_model(vec2text_model)
    if 'Zuco' not in args.dataset:
        args.target_lang = 'zh_CN'
    else:
        args.target_lang = 'EN'
        
                                                         
    for param in corrector.model.parameters():
        param.requires_grad = True

                           
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

                                           
    experiment_dir = os.path.join("results", args.dataset)
    embedding_dir = os.path.join(experiment_dir, "embeddings")
    model_dir = os.path.join(experiment_dir, "model")
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    best_eval_loss = float('inf')
    best_cosine_similarity = float('-inf')
    stop_epoch = 0

                   
    for epoch in range(1, args.num_epochs + 1):
                                  
        train_loss = train_neural_encoder(
            model, train_loader, model_optimizer, loss_fn, device, epoch, embedding_dir
        )

                            
        val_loss, val_cosine_similarity, all_neural_embeddings, all_text_embeddings, avg_bert_score, original_sens, pred_sens = evaluate(
            args,
            model,
            val_loader,
            loss_fn,
            device,
            epoch,
            corrector,
            similarity_model,
            n_steps=args.n_steps,
            save_embeddings=True,
            save_freq=10,
            output_dir=embedding_dir
        )

                                                
                                       
                                       
                                                                                                 
                                                                                                                                 
                                                                                                              
                                                                                             

        if val_cosine_similarity > best_cosine_similarity:
            stop_epoch = 0
            best_cosine_similarity = val_cosine_similarity
            torch.save(model.state_dict(), os.path.join(model_dir, f"best_cosine_similarity_model_{args.task}_{args.subjects}.pth"))
            np.save(os.path.join(embedding_dir, f"best_cosine_similarity_neural_embeddings_{args.task}_{args.subjects}.npy"), np.concatenate(all_neural_embeddings))
            with open(os.path.join(experiment_dir, f'result_{args.task}_{args.subjects}.json'), 'w') as f:
                json.dump({'orig_texts': original_sens, 'pred_texts': pred_sens}, f, indent=4)
            print(f"Saved model and embeddings with best cosine similarity at epoch {epoch}")
        else:
            stop_epoch +=1
            if stop_epoch > args.early_stop:
                break

    return best_cosine_similarity
                                     
                                                                                                                      

                                              
    optimizer = torch.optim.AdamW(corrector.model.parameters(), lr=args.corrector_lr)
    best_bert_score = float('-inf')

    for epoch in range(1, 3):
        fine_tune_corrector(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            corrector
        )
        val_loss, val_cosine_similarity, all_neural_embeddings, all_text_embeddings, avg_bert_score, original_sens, pred_sens = evaluate(
            args,
            model,
            val_loader,
            loss_fn,
            device,
            epoch,
            corrector,
            similarity_model,
            n_steps=args.n_steps,
            save_embeddings=True,
            save_freq=10,
            output_dir=embedding_dir
        )

        if avg_bert_score > best_bert_score:
            best_bert_score = avg_bert_score
            torch.save(corrector.model.state_dict(), os.path.join(model_dir, f"best_corrector_bert_score_model_{args.task}_{args.subjects}.pth"))
            np.save(os.path.join(embedding_dir, f"best_bert_score_neural_embeddings_{args.task}_{args.subjects}.npy"), np.concatenate(all_neural_embeddings))
            np.save(os.path.join(embedding_dir, f"best_bert_score_text_embeddings_{args.task}_{args.subjects}.npy"), np.concatenate(all_text_embeddings))
            print(f"Saved corrector model and embeddings with best BERT score at epoch {epoch}")
        

def init_csv_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'mean_cosine'])

def append_to_csv(file_path, dataset_name, mean_cosine):
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([dataset_name, mean_cosine])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Encoder with CLIP Loss")
    parser.add_argument(
        '--file_path',
        type=str,
        default='',
        help='Path to the data file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Zuco_1',
        help='Datasets of downstream tasks'
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="ZAB",
        help="List of subjects"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="task1-SR",
        help="List of subjects"
    )
    parser.add_argument(
        "--bands",
        nargs='+',
        default=["highgamma"],
        help="List of frequency bands"
    )
    parser.add_argument(
        "--level",
        type=str,
        default="sentence",
        help="Level of processing: word, sentence, or custom"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Padding around segments"
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=10,
        help="N-gram size for custom level"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1,
        help="Number of steps for reconstruction"
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="gtr-t5-base",                           
        help="Embedding model to use"
    )
                          
                            
                   
                                   
                                   
       
                          
                             
                   
                                
                               
       
    parser.add_argument(
        "--leave_out_trials",
        nargs='+',
        type=int,
        default=[0],
        help="Indices of trials to leave out for evaluation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0013,
        help="Learning rate for the neural encoder"
    )
    parser.add_argument(
        "--corrector_lr",
        type=float,
        default=0.0013,
        help="Learning rate for the corrector model"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=40,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=6,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature parameter for CLIP loss"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10,
        help="Frequency (in epochs) to save model checkpoints"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for n-gram processing"
    )
    parser.add_argument(
        '--vec2text_model_name',
        type=str,
        default="gtr-base",
        help="Vec2Text model to use for reconstruction"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=2.0,
        help="Margin for Triplet Margin Loss"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Alpha parameter for combining CLIP and Triplet losses"
    )

    args = parser.parse_args()

    best_cosine_similarity = main(args)
    
    if not os.path.exists('ans_LLM.csv'):
        init_csv_file('ans_LLM.csv')
    dataset_name = f"{args.dataset}_{args.subjects}_{args.task}"
    append_to_csv('ans_LLM.csv', dataset_name, best_cosine_similarity)
