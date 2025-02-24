import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mteb import MTEB
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from prune_frida_heads import prune_model
from evaluate_frida_v2 import PromptType, FridaModel

class PrunedFridaModel(FridaModel):
    def __init__(self, model_name="ai-forever/FRIDA"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Initializing model with pruned attention heads...")
        self.model = prune_model(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.layer_activations = []

def main():
    print("Initializing FRIDA model with pruned attention heads...")
    model = PrunedFridaModel()
    
    print("\nRunning evaluation...")
    evaluation = MTEB(tasks=["GeoreviewClusteringP2P"])
    results = evaluation.run(model, output_folder="results_pruned")
    
    print("\nEvaluation Results:")
    print(results)
    
    print("\nAnalyzing layer activations...")
    stats = model.get_layer_statistics()
    
    print("\nLayer Activation Statistics:")
    for layer_idx in range(len(stats["mean"])):
        print(f"\nLayer {layer_idx}:")
        print(f"Mean: {stats['mean'][layer_idx]:.4f}")
        print(f"Std:  {stats['std'][layer_idx]:.4f}")
        print(f"Max:  {stats['max'][layer_idx]:.4f}")
        print(f"Min:  {stats['min'][layer_idx]:.4f}")
    
    model.plot_layer_statistics(stats, save_path="layer_activations_pruned.png")
    print("\nLayer activation plot saved as layer_activations_pruned.png")

if __name__ == "__main__":
    main()