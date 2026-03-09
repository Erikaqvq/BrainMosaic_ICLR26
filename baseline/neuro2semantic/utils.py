import torch
from typing import List
import bert_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import string
import pickle
from torch.utils.data import DataLoader, random_split
from models import NeuralEncoder
import vec2text

def calculate_bert_score(reconstructed_texts: List[str], original_texts: List[str]) -> tuple:
    """
    Calculate the BERT score between reconstructed and original texts.

    Args:
        reconstructed_texts (List[str]): List of reconstructed text strings.
        original_texts (List[str]): List of original text strings.

    Returns:
        tuple: Average precision, recall, and F1 scores.
    """
    if isinstance(reconstructed_texts, tuple):
        reconstructed_texts = list(reconstructed_texts)
    if isinstance(original_texts, tuple):
        original_texts = list(original_texts)
    
    P, R, F1 = bert_score.score(
        reconstructed_texts, original_texts, lang="en", rescale_with_baseline=True
    )
    return P.mean().item(), R.mean().item(), F1.mean().item()




def calculate_rouge(reconstructed_texts: List[str], original_texts: List[str]) -> float:
    """
    Calculate the ROUGE-1 F1 score between reconstructed and original texts.

    Args:
        reconstructed_texts (List[str]): List of reconstructed text strings.
        original_texts (List[str]): List of original text strings.

    Returns:
        float: Average ROUGE-1 F1 score.
    """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(reconstructed_texts, original_texts, avg=True)
    return rouge_scores['rouge-1']['f']



def calculate_bleu_score(reconstructed_texts: List[str], original_texts: List[str]) -> float:
    """
    Calculate the BLEU score between reconstructed texts and original texts.

    Args:
        reconstructed_texts (List[str]): List of reconstructed text strings.
        original_texts (List[str]): List of original text strings.

    Returns:
        float: Average BLEU score.
    """
    smoothie = SmoothingFunction().method7                                                
    bleu_scores = []

    for recon, orig in zip(reconstructed_texts, original_texts):
        recon = recon.translate(str.maketrans('', '', string.punctuation))
        orig = orig.translate(str.maketrans('', '', string.punctuation))

        reference = [orig.split()]
        candidate = recon.split()
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        bleu_scores.append(bleu_score)

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0



def load_neural_encoder(model_path: str, input_dim: int, device: torch.device) -> NeuralEncoder:
    """
    Load the trained NeuralEncoder model.

    Args:
        model_path (str): Path to the trained model weights.
        input_dim (int): Dimension of the input features.
        device (torch.device): Device to load the model on.

    Returns:
        NeuralEncoder: Loaded NeuralEncoder model.
    """
    model = NeuralEncoder(input_dim=input_dim, embedding_dim=768)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_vec2text_model(vec2text_model_name: str) -> vec2text.trainers.Corrector:
    """
    Load the Vec2Text corrector model.

    Args:
        vec2text_model_name (str): Name of the Vec2Text model to load.

    Returns:
        vec2text.trainers.Corrector: Loaded Vec2Text corrector model.
    """
    corrector = vec2text.load_pretrained_corrector(vec2text_model_name)
    return corrector