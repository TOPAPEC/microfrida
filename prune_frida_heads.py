import torch
from transformers import T5EncoderModel
from typing import List, Tuple, Dict
import numpy as np
from inspect_frida_model import calculate_attention_head_importance
import json

def get_heads_to_prune(importance_matrix: np.ndarray) -> List[Tuple[int, List[int]]]:
    num_layers = importance_matrix.shape[0]
    heads_to_prune = []
    
    for layer_idx in range(num_layers):
        layer_importance = importance_matrix[layer_idx]
        num_heads_to_prune = 20
            
        head_indices = np.argsort(layer_importance)[:num_heads_to_prune]
        heads_to_prune.append((layer_idx, head_indices.tolist()))
    
    return heads_to_prune

def prune_heads(model: T5EncoderModel, heads_to_prune: List[Tuple[int, List[int]]]):
    for layer_idx, head_indices in heads_to_prune:
        layer = model.encoder.block[layer_idx].layer[0].SelfAttention
        head_size = model.config.hidden_size // model.config.num_heads
        
        for head_idx in head_indices:
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            
            with torch.no_grad():
                layer.q.weight[start_idx:end_idx, :] = 0
                layer.k.weight[start_idx:end_idx, :] = 0
                layer.v.weight[start_idx:end_idx, :] = 0
                if hasattr(layer.q, 'bias') and layer.q.bias is not None:
                    layer.q.bias[start_idx:end_idx] = 0
                    layer.k.bias[start_idx:end_idx] = 0
                    layer.v.bias[start_idx:end_idx] = 0

def prune_model(model_name: str = "ai-forever/FRIDA") -> Dict:
    print(f"Loading model {model_name}...")
    model = T5EncoderModel.from_pretrained(model_name)
    
    print("\nCalculating attention head importance...")
    importance_matrix = calculate_attention_head_importance(model)
    
    print("\nDetermining heads to prune...")
    heads_to_prune = get_heads_to_prune(importance_matrix)
    
    print("\nPruning 12 least important heads in every layer...")
    prune_heads(model, heads_to_prune)
    
    pruning_info = {
        "num_layers": len(model.encoder.block),
        "pruned_heads": [
            {"layer": layer, "heads": heads} 
            for layer, heads in heads_to_prune
        ]
    }
    
    with open("pruning_results.json", "w") as f:
        json.dump(pruning_info, f, indent=4)
    print("\nPruning results saved to pruning_results.json")
    
    return model

if __name__ == "__main__":
    pruned_model = prune_model()