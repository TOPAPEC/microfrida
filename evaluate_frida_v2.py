import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
from mteb import MTEB, get_benchmark
from tqdm import tqdm
from enum import Enum
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

class PromptType(str, Enum):
    query = "query"
    passage = "passage"

class FridaModel:
    frida_prompts = {
        "Classification": "categorize: ",
        "MultilabelClassification": "categorize: ",
        "Clustering": "categorize_topic: ",
        "PairClassification": "paraphrase: ",
        "Reranking": "paraphrase: ",
        f"Reranking-{PromptType.query}": "search_query: ",
        f"Reranking-{PromptType.passage}": "search_document: ",
        "STS": "paraphrase: ",
        "Summarization": "categorize: ",
        PromptType.query: "search_query: ",
        PromptType.passage: "search_document: ",
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
        "RiaNewsRetrieval": "categorize: ",
    }

    def __init__(self, model_name="ai-forever/FRIDA"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.layer_activations = []
        self.current_batch_size = 0

    def pool(self, hidden_state, mask, pooling_method="cls"):
        if pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif pooling_method == "cls":
            return hidden_state[:, 0]

    def calculate_layer_norms(self, hidden_states: Tuple[torch.Tensor]) -> List[float]:
        norms = []
        for layer_output in hidden_states[1:]:  
            layer_norm = torch.norm(layer_output, dim=-1).mean().item()
            norms.append(layer_norm)
        return norms

    def update_layer_statistics(self, layer_norms: List[float]):
        if not self.layer_activations:
            self.layer_activations = [[] for _ in range(len(layer_norms))]
        
        for layer_idx, norm in enumerate(layer_norms):
            self.layer_activations[layer_idx].append(norm)

    def get_layer_statistics(self) -> Dict[str, List[float]]:
        stats = {
            "mean": [],
            "std": [],
            "max": [],
            "min": []
        }
        
        for layer_acts in self.layer_activations:
            layer_acts = np.array(layer_acts)
            stats["mean"].append(float(np.mean(layer_acts)))
            stats["std"].append(float(np.std(layer_acts)))
            stats["max"].append(float(np.max(layer_acts)))
            stats["min"].append(float(np.min(layer_acts)))
            
        return stats

    def plot_layer_statistics(self, stats: Dict[str, List[float]], save_path: Optional[str] = None):
        plt.figure(figsize=(12, 6))
        layers = range(len(stats["mean"]))
        
        plt.plot(layers, stats["mean"], 'b-', label='Mean Activation')
        plt.fill_between(layers,
                        np.array(stats["mean"]) - np.array(stats["std"]),
                        np.array(stats["mean"]) + np.array(stats["std"]),
                        alpha=0.2)
        
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
            batch = sentences[i:i + batch_size]
            batch = [f"{prompt}{sent}" for sent in batch]
            
            encoded = self.tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=True)
                
                layer_norms = self.calculate_layer_norms(outputs.hidden_states)
                self.update_layer_statistics(layer_norms)
                
                embeddings = self.pool(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                    pooling_method="cls"
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0)

def main():
    model = FridaModel()
    
    evaluation = MTEB(tasks=["GeoreviewClusteringP2P"])
    results = evaluation.run(model, output_folder="results")
    
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
    
    model.plot_layer_statistics(stats, save_path="layer_activations.png")
    print("\nLayer activation plot saved as layer_activations.png")

if __name__ == "__main__":
    main()