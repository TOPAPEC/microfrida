#!/usr/bin/env python3
import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import T5EncoderModel, AutoTokenizer
from mteb import MTEB

def evaluate_model(model_dir, tasks=["GeoreviewClusteringP2P"], eval_output_folder="eval_results"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5EncoderModel.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class FridaModel:
        frida_prompts = {
            "Classification": "categorize: ",
            "MultilabelClassification": "categorize: ",
            "Clustering": "categorize_topic: ",
            "PairClassification": "paraphrase: ",
            "Reranking": "paraphrase: ",
            "Reranking-query": "search_query: ",
            "Reranking-passage": "search_document: ",
            "STS": "paraphrase: ",
            "Summarization": "categorize: ",
            "query": "search_query: ",
            "passage": "search_document: ",
            "CEDRClassification": "categorize_sentiment: ",
            "GeoreviewClassification": "categorize_sentiment: ",
            "HeadlineClassification": "categorize_topic: ",
            "InappropriatenessClassification": "categorize_topic: ",
            "KinopoiskClassification": "categorize_sentiment: ",
            "MassiveIntentClassification": "paraphrase: ",
            "MassiveScenarioClassification": "paraphrase: ",
            "RuReviewsClassification": "categorize_sentiment: ",
            "RuSciBenchGRNTIClassification": "categorize_topic: ",
            "RuSciBenchOECDClassification": "categorize_topic: ",
            "SensitiveTopicsClassification": "categorize_topic: ",
            "TERRa": "categorize_entailment: ",
            "RiaNewsRetrieval": "categorize: "
        }
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.layer_activations = []
        def pool(self, hidden_state, mask, pooling_method="cls"):
            if pooling_method == "mean":
                s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
                d = mask.sum(axis=1, keepdim=True).float()
                return s / d
            elif pooling_method == "cls":
                return hidden_state[:, 0]
        def calculate_layer_norms(self, hidden_states):
            norms = []
            for layer_output in hidden_states[1:]:
                layer_norm = torch.norm(layer_output, dim=-1).mean().item()
                norms.append(layer_norm)
            return norms
        def update_layer_statistics(self, layer_norms):
            if not self.layer_activations:
                self.layer_activations = [[] for _ in range(len(layer_norms))]
            for layer_idx, norm in enumerate(layer_norms):
                self.layer_activations[layer_idx].append(norm)
        def get_layer_statistics(self):
            stats = {"mean": [], "std": [], "max": [], "min": []}
            for layer_acts in self.layer_activations:
                layer_acts = np.array(layer_acts)
                stats["mean"].append(float(np.mean(layer_acts)))
                stats["std"].append(float(np.std(layer_acts)))
                stats["max"].append(float(np.max(layer_acts)))
                stats["min"].append(float(np.min(layer_acts)))
            return stats
        def plot_layer_statistics(self, stats, save_path=None):
            plt.figure(figsize=(12, 6))
            layers = range(len(stats["mean"]))
            plt.plot(layers, stats["mean"], 'b-', label='Mean Activation')
            plt.fill_between(layers, np.array(stats["mean"]) - np.array(stats["std"]), np.array(stats["mean"]) + np.array(stats["std"]), alpha=0.2)
            plt.plot(layers, stats["max"], 'r--', label='Max Activation')
            plt.plot(layers, stats["min"], 'g--', label='Min Activation')
            plt.xlabel('Layer Index')
            plt.ylabel('Activation Norm')
            plt.title('Layer-wise Activation Statistics')
            plt.legend()
            plt.grid(True)
            if save_path:
                plt.savefig(save_path)
            plt.close()
        def encode(self, sentences, batch_size=128, task_type=None, **kwargs):
            all_embeddings = []
            prompt = self.frida_prompts.get(task_type, "categorize: ")
            for i in tqdm(range(0, len(sentences), batch_size), desc="Evaluating"):
                batch = sentences[i:i+batch_size]
                batch = [f"{prompt}{sent}" for sent in batch]
                encoded = self.tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.no_grad():
                    outputs = self.model(**encoded, output_hidden_states=True)
                    layer_norms = self.calculate_layer_norms(outputs.hidden_states)
                    self.update_layer_statistics(layer_norms)
                    embeddings = self.pool(outputs.last_hidden_state, encoded["attention_mask"], pooling_method="cls")
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.cpu())
            return torch.cat(all_embeddings, dim=0)
    frida_model = FridaModel(model, tokenizer, device)
    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(frida_model, output_folder=eval_output_folder)
    stats = frida_model.get_layer_statistics()
    frida_model.plot_layer_statistics(stats, save_path=f"{eval_output_folder}/layer_activations.png")
    return results, stats

if __name__ == "__main__":
    # Optionally, add code to invoke evaluation here.
    pass
