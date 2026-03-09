import argparse

if __name__ == "__main__":
    _parser = argparse.ArgumentParser("neuro2semantic reconstruction (public release)")
    _parser.add_argument("--config", required=True, help="Path to baseline config JSON.")
    _parser.parse_args()
    raise RuntimeError(
        "This reconstruction baseline is intentionally disabled in the public release."
    )

import torch
import numpy as np
import vec2text
from torch.utils.data import DataLoader, random_split
import os
import pickle
from data_loader import create_dataloader, NeuralDataset
from typing import List
from utils import load_neural_encoder, load_vec2text_model

def reconstruct_text(
    embeddings: torch.Tensor,
    corrector: vec2text.trainers.Corrector,
    n_steps: int = 4,
    seq_beam_width: int = 4,
    target_lang: str = 'zh_CN',
    is_train: bool = True
) -> List[str]:
    """
    Reconstruct text from embeddings using the Vec2Text corrector.

    Args:
        embeddings (torch.Tensor): Embeddings to invert.
        corrector (vec2text.trainers.Corrector): Vec2Text corrector model.
        n_steps (int, optional): Number of recursive steps. Defaults to 4.
        seq_beam_width (int, optional): Beam width for sequence generation. Defaults to 4.

    Returns:
        List[str]: List of reconstructed text strings.
    """
    kwargs = {}
    
    if n_steps is not None:
        kwargs['num_steps'] = n_steps
    if seq_beam_width is not None:
        kwargs['sequence_beam_width'] = seq_beam_width
    
    if 'CN' in target_lang:
        reconstructed_text = custom_invert_embeddings(embeddings=embeddings.cuda(),
                                corrector=corrector,
                                target_lang=target_lang, training=is_train
                                )
    else:
        vec2text.trainers.Corrector.invert_embeddings = custom_invert_embeddings

        reconstructed_text = vec2text.invert_embeddings(
            embeddings=embeddings.cuda(),
            corrector=corrector,
            **kwargs 
        )
    
    return reconstructed_text


def custom_invert_embeddings(
    embeddings: torch.Tensor,
    corrector: vec2text.trainers.Corrector,
    num_steps: int = None,
    sequence_beam_width: int = 0,
    training: bool = True,
    target_lang: str = 'zh_CN'
) -> List[str]:
    """
    Custom method to invert embeddings using the Vec2Text corrector.

    Args:
        self: Placeholder for method signature compatibility.
        embeddings (torch.Tensor): Embeddings to invert.
        corrector (vec2text.trainers.Corrector): Vec2Text corrector model.
        num_steps (int, optional): Number of recursive steps. Defaults to None.
        sequence_beam_width (int, optional): Beam width for sequence generation. Defaults to 0.
        training (bool, optional): Flag indicating training mode. Defaults to True.

    Returns:
        List[str]: List of regenerated text strings.
    """
    if not training:
        corrector.inversion_trainer.model.eval()
        corrector.model.eval()
    else:
        corrector.inversion_trainer.model.train()
        corrector.model.train()

    gen_kwargs = corrector.gen_kwargs.copy()
    gen_kwargs["min_length"] = 1
    gen_kwargs["max_length"] = 128
    
    if hasattr(corrector.tokenizer, "lang_code_to_id") and target_lang in corrector.tokenizer.lang_code_to_id:
        gen_kwargs["forced_bos_token_id"] = corrector.tokenizer.lang_code_to_id[target_lang]           
    else:
                                          
        prompt = f"Answer in {target_lang}:"

    if num_steps is None:
        assert (
            sequence_beam_width == 0
        ), "Cannot set a nonzero beam width without multiple steps"

        regenerated = corrector.inversion_trainer.generate(
            inputs={
                "frozen_embeddings": embeddings,
                "prompt": [prompt] * embeddings.size(0),                
            },
            generation_kwargs=gen_kwargs,
        )
    else:
        corrector.return_best_hypothesis = sequence_beam_width > 0
        regenerated = corrector.generate(
            inputs={
                "frozen_embeddings": embeddings,
                "prompt": [prompt] * embeddings.size(0),                
            },
            generation_kwargs=gen_kwargs,
            num_recursive_steps=num_steps,
            sequence_beam_width=sequence_beam_width,
        )

    output_strings = corrector.tokenizer.batch_decode(
        regenerated, skip_special_tokens=True
    )
    return output_strings


def main(args):
    """
    Main function to reconstruct text from neural embeddings.

    Args:
        args: Parsed command-line arguments.
    """
                
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
               
    dataloader, input_dim = create_dataloader(
        args.file_path,
        subjects=args.subjects,
        level=args.level,
        bands=args.bands,
        padding=args.padding,
        n=args.ngram,
        embedding_model_name=args.embedding_model_name
    )
    
                                                  
    dataset_size = len(dataloader.dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    _, val_dataset = random_split(dataloader.dataset, [train_size, val_size])
    
                                  
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataloader.collate_fn
    )

                               
    model = load_neural_encoder(args.model_path, input_dim, device)

                         
    corrector = load_vec2text_model(args.vec2text_model_name)

                                                   
    output_dir = os.path.join("results", args.wandb_run_name, "reconstructed_text")
    os.makedirs(output_dir, exist_ok=True)
    
                                                       
    reconstructions = []

    with torch.no_grad():
        for neural_segment, text_embedding, masks, original_text in val_loader:
            neural_segment = neural_segment.to(device)
            text_embedding = text_embedding.to(device)
            masks = masks.to(device)
            neural_features = model(neural_segment, masks)
            
                                                                               
            reconstructed_text_from_text = reconstruct_text(text_embedding, corrector)
            
                                                                        
            reconstructed_text_from_neural = reconstruct_text(neural_features, corrector)

            for i in range(len(reconstructed_text_from_text)):
                reconstructions.append({
                    'original_text': original_text[i],
                    'reconstructed_text_from_text': reconstructed_text_from_text[i],
                    'reconstructed_text_from_neural': reconstructed_text_from_neural[i]
                })
    
                                        
    reconstruction_file = os.path.join(output_dir, "reconstructions.pkl")
    with open(reconstruction_file, "wb") as f:
        pickle.dump(reconstructions, f)
    
    print(f"Reconstructed texts saved to {reconstruction_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct Text from Neural Embeddings using Vec2Text"
    )
    parser.add_argument(
        '--file_path',
        type=str,
        default='path/to/data_all.pkl',
        help='Path to the data file'
    )
    parser.add_argument(
        "--subjects",
        nargs='+',
        default=["Subject1"],
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
        default="custom",
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
        default=3,
        help="N-gram size for custom level"
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="gtr-t5-base",
        help="Embedding model to use"
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        required=True,
        help='WandB run name'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained neural encoder model'
    )
    parser.add_argument(
        '--vec2text_model_name',
        type=str,
        default="gtr-base",
        help="Vec2Text model to use for reconstruction"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing validation data"
    )

    args = parser.parse_args()
    
    main(args)
