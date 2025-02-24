import torch
from transformers import AutoTokenizer, T5EncoderModel
from typing import Dict
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_attention_head_importance(model: T5EncoderModel) -> np.ndarray:
    num_layers = len(model.encoder.block)
    num_heads = model.config.num_heads
    importance_matrix = np.zeros((num_layers, num_heads))
    
    for layer_idx in range(num_layers):
        layer = model.encoder.block[layer_idx].layer[0].SelfAttention
        # Get query, key, value weights
        q_weights = layer.q.weight.detach()
        k_weights = layer.k.weight.detach()
        v_weights = layer.v.weight.detach()
        
        # Reshape weights to [num_heads, head_dim, hidden_dim]
        head_dim = model.config.hidden_size // num_heads
        q_heads = q_weights.view(num_heads, head_dim, -1)
        k_heads = k_weights.view(num_heads, head_dim, -1)
        v_heads = v_weights.view(num_heads, head_dim, -1)
        
        # Calculate importance score for each head
        for head_idx in range(num_heads):
            # Compute norm of the combined QKV weights for each head
            head_norm = (torch.norm(q_heads[head_idx]) * 
                        torch.norm(k_heads[head_idx]) * 
                        torch.norm(v_heads[head_idx]))
            importance_matrix[layer_idx, head_idx] = head_norm.item()
    
    # Normalize importance scores
    importance_matrix = (importance_matrix - importance_matrix.min()) / (importance_matrix.max() - importance_matrix.min())
    return importance_matrix

def plot_attention_importance(importance_matrix: np.ndarray, save_path: str = "attention_importance.png"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(importance_matrix, 
                cmap='viridis', 
                xticklabels=[f'Head {i+1}' for i in range(importance_matrix.shape[1])],
                yticklabels=[f'Layer {i+1}' for i in range(importance_matrix.shape[0])],
                annot=True, 
                fmt='.2f')
    plt.title('Attention Head Importance Across Layers')
    plt.xlabel('Attention Heads')
    plt.ylabel('Layers')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nAttention importance heatmap saved to {save_path}")

def inspect_model(model_name: str = "ai-forever/FRIDA") -> Dict:
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    
    model_info = {}
    
    # Get number of layers
    num_layers = len(model.encoder.block)
    model_info["num_layers"] = num_layers
    print(f"\nNumber of layers: {num_layers}")
    
    # Get hidden state size
    hidden_size = model.config.hidden_size
    model_info["hidden_size"] = hidden_size
    print(f"Hidden state size: {hidden_size}")
    
    # Check for LoRA
    has_lora = False
    for name, module in model.named_modules():
        if 'lora' in name.lower():
            has_lora = True
            break
    model_info["has_lora"] = has_lora
    print(f"LoRA detected: {has_lora}")
    
    # Additional model configuration details
    print("\nAdditional model configuration:")
    print(f"Feed forward dimension: {model.config.d_ff}")
    print(f"Number of attention heads: {model.config.num_heads}")
    print(f"Dropout rate: {model.config.dropout_rate}")
    
    model_info["feed_forward_dim"] = model.config.d_ff
    model_info["num_attention_heads"] = model.config.num_heads
    model_info["dropout_rate"] = model.config.dropout_rate
    
    # Calculate and plot attention head importance
    print("\nCalculating attention head importance...")
    importance_matrix = calculate_attention_head_importance(model)
    plot_attention_importance(importance_matrix)
    
    # Add attention importance to model info
    model_info["attention_importance"] = importance_matrix.tolist()
    
    # Save model info to file
    with open("model_inspection_results.json", "w") as f:
        json.dump(model_info, f, indent=4)
    print("\nDetailed results saved to model_inspection_results.json")
    
    return model_info

if __name__ == "__main__":
    inspect_model()