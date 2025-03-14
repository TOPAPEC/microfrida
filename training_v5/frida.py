import torch
from transformers import AutoTokenizer, T5EncoderModel

class FridaModelAccess:
    def __init__(self, model_name="ai-forever/FRIDA", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.encoder = self.model.encoder
        self.last_layer = self.encoder.block[-1]
        self.self_attention = self.last_layer.layer[0].SelfAttention
    
    def get_embedding_matrix(self):
        return self.model.shared.weight
    
    def process_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
            
            last_hidden_state = outputs.last_hidden_state
            
            q = self.self_attention.q(last_hidden_state)
            k = self.self_attention.k(last_hidden_state)
            v = self.self_attention.v(last_hidden_state)
            
            attention_weights = outputs.attentions[-1] if outputs.attentions else None
        
        return {
            "last_hidden_state": last_hidden_state,
            "q": q,
            "k": k,
            "v": v,
            "attention_weights": attention_weights,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    
    def get_qkv_matrices(self, text):
        result = self.process_text(text)
        return {
            "q": result["q"],
            "k": result["k"],
            "v": result["v"]
        }
    
    def get_last_hidden_state(self, text):
        result = self.process_text(text)
        return result["last_hidden_state"]
    
    def get_attention_weights(self, text):
        result = self.process_text(text)
        return result["attention_weights"]
