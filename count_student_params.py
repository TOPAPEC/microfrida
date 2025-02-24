#!/usr/bin/env python3
from transformers import T5EncoderModel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def count_parameters(model):
    """Return the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def gather_component_params(student_model):
    # Count parameters in embeddings, each encoder block, and final layer norm:
    embedding_params = sum(p.numel() for p in student_model.shared.parameters())
    layer_param_counts = [sum(p.numel() for p in layer.parameters())
                          for layer in student_model.encoder.block]
    final_layer_norm_params = sum(p.numel() for p in student_model.encoder.final_layer_norm.parameters())
    
    component_names = (
        ["Embeddings"] +
        [f"Layer {i}" for i in range(len(layer_param_counts))] +
        ["Final Layer Norm"]
    )
    param_counts = [embedding_params] + layer_param_counts + [final_layer_norm_params]
    
    return component_names, param_counts

def plot_parameter_distribution(components, counts):
    # Plot horizontal bar chart:
    plt.figure(figsize=(10, 6))
    plt.barh(components, counts, color="skyblue")
    plt.xlabel("Number of Parameters")
    plt.title("Parameter Distribution in Student Model")
    plt.tight_layout()
    bar_chart_path = "parameter_distribution_bar.png"
    plt.savefig(bar_chart_path)
    plt.close()
    
    # Plot heatmap:
    # Create a 2D array (column vector) for heatmap visualization.
    param_array = np.array(counts).reshape(-1, 1)
    plt.figure(figsize=(6, 8))
    sns.heatmap(param_array, annot=True, fmt="d", cmap="YlGnBu",
                yticklabels=components, xticklabels=["Param Count"])
    plt.title("Parameter Distribution Heatmap")
    plt.tight_layout()
    heatmap_path = "parameter_distribution_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    
    return bar_chart_path, heatmap_path

def main():
    model_path = "./student_FRIDA"
    # Load the student model.
    student_model = T5EncoderModel.from_pretrained(model_path)
    
    # Count total parameters.
    total_params = count_parameters(student_model)
    print("Total number of parameters in the student model:", total_params)
    
    # Gather parameters from model components.
    component_names, param_counts = gather_component_params(student_model)
    print("Parameter distribution:")
    for comp, cnt in zip(component_names, param_counts):
        print(f"{comp}: {cnt}")
    
    # Generate and save visualizations.
    bar_chart_path, heatmap_path = plot_parameter_distribution(component_names, param_counts)
    print(f"Saved bar chart as '{bar_chart_path}' and heatmap as '{heatmap_path}'.")

if __name__ == "__main__":
    main()
