import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
from mteb import MTEB, get_benchmark
from tqdm import tqdm
from enum import Enum

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

    def pool(self, hidden_state, mask, pooling_method="cls"):
        if pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif pooling_method == "cls":
            return hidden_state[:, 0]

    def encode(self, sentences, batch_size=128, task_type=None, **kwargs):
        all_embeddings = []
        prompt = self.frida_prompts.get(task_type, "categorize: ")
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Evaluating"):
            batch = sentences[i:i + batch_size]
            batch = [f"{prompt}{sent}" for sent in batch]
            
            encoded = self.tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
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
    
    evaluation = MTEB(tasks=["GeoreviewClusteringP2P", "RuSciBenchGRNTIClusteringP2P", "RuSciBenchOECDClusteringP2P"])
    results = evaluation.run(model, output_folder="results")
    
    print("Evaluation Results:")
    print(results)

if __name__ == "__main__":
    main()