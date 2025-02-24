#!/usr/bin/env python3
import os
import copy
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import T5EncoderModel, T5Config, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from transformers.models.t5.modeling_t5 import T5Attention
from datasets import load_dataset, Dataset
from torch.distributed.elastic.multiprocessing.errors import record
from mteb import MTEB
from torch.nn.parallel import DistributedDataParallel as DDP

def prune_linear_layer_generic(linear_layer, top_indices, num_heads_teacher, head_dim):
    d_model = linear_layer.in_features
    w = linear_layer.weight.data.t().contiguous().view(d_model, num_heads_teacher, head_dim)
    pruned_w = w[:, top_indices, :].contiguous().view(d_model, len(top_indices) * head_dim)
    pruned_w = pruned_w.t().contiguous()
    pruned_b = None
    if linear_layer.bias is not None:
        b = linear_layer.bias.data.view(num_heads_teacher, head_dim)
        pruned_b = b[top_indices, :].contiguous().view(len(top_indices) * head_dim)
    return pruned_w, pruned_b

def prune_linear_layer_o(o_layer, top_indices, num_heads_teacher, head_dim):
    d_model = o_layer.weight.data.size(0)
    w = o_layer.weight.data.t().contiguous().view(num_heads_teacher, head_dim, d_model)
    pruned_w = w[top_indices, :, :].contiguous().view(len(top_indices) * head_dim, d_model)
    pruned_w = pruned_w.t().contiguous()
    pruned_b = None
    if o_layer.bias is not None:
        pruned_b = o_layer.bias.data
    return pruned_w, pruned_b

def select_top_heads(linear_layer, num_heads_teacher, head_dim, new_num_heads):
    d_model = linear_layer.in_features
    w = linear_layer.weight.data.t().contiguous().view(d_model, num_heads_teacher, head_dim)
    head_norms = w.pow(2).sum(dim=(0, 2)).sqrt()
    top_indices = torch.topk(head_norms, new_num_heads).indices
    return torch.sort(top_indices).values

def prune_attention_module(attn_module, new_num_heads, student_config):
    num_heads_teacher = attn_module.n_heads
    d_model = attn_module.d_model
    head_dim = d_model // num_heads_teacher
    q_layer = attn_module.q
    k_layer = attn_module.k
    v_layer = attn_module.v
    o_layer = attn_module.o
    top_indices = select_top_heads(q_layer, num_heads_teacher, head_dim, new_num_heads)
    q_w, q_b = prune_linear_layer_generic(q_layer, top_indices, num_heads_teacher, head_dim)
    k_w, k_b = prune_linear_layer_generic(k_layer, top_indices, num_heads_teacher, head_dim)
    v_w, v_b = prune_linear_layer_generic(v_layer, top_indices, num_heads_teacher, head_dim)
    o_w, o_b = prune_linear_layer_o(o_layer, top_indices, num_heads_teacher, head_dim)
    new_attn = T5Attention(student_config, has_relative_attention_bias=attn_module.has_relative_attention_bias)
    new_attn.q = nn.Linear(d_model, new_num_heads * head_dim, bias=(q_b is not None))
    new_attn.k = nn.Linear(d_model, new_num_heads * head_dim, bias=(k_b is not None))
    new_attn.v = nn.Linear(d_model, new_num_heads * head_dim, bias=(v_b is not None))
    new_attn.o = nn.Linear(new_num_heads * head_dim, d_model, bias=(o_b is not None))
    new_attn.q.weight.data.copy_(q_w)
    new_attn.k.weight.data.copy_(k_w)
    new_attn.v.weight.data.copy_(v_w)
    new_attn.o.weight.data.copy_(o_w)
    if q_b is not None:
        new_attn.q.bias.data.copy_(q_b)
        new_attn.k.bias.data.copy_(k_b)
        new_attn.v.bias.data.copy_(v_b)
        new_attn.o.bias.data.copy_(o_b)
    new_attn.n_heads = new_num_heads
    new_attn.d_model = d_model
    new_attn.dropout = student_config.dropout_rate
    return new_attn

def prune_t5_block(teacher_block, new_num_heads, student_config):
    student_block = copy.deepcopy(teacher_block)
    student_block.layer[0].SelfAttention = prune_attention_module(teacher_block.layer[0].SelfAttention, new_num_heads=new_num_heads, student_config=student_config)
    return student_block

def build_student_model(teacher_model, new_num_heads, blocks_to_keep_indices):
    teacher_config = teacher_model.config
    student_config_dict = teacher_config.to_dict()
    student_config_dict["num_layers"] = len(blocks_to_keep_indices)
    student_config_dict["num_heads"] = new_num_heads
    student_config = teacher_config.__class__(**student_config_dict)
    student_model = T5EncoderModel(student_config)
    student_model.shared = teacher_model.shared
    student_model.encoder.set_input_embeddings(teacher_model.shared)
    teacher_blocks = teacher_model.encoder.block
    student_blocks = []
    for idx in blocks_to_keep_indices:
        teacher_block = teacher_blocks[idx]
        pruned_block = prune_t5_block(teacher_block, new_num_heads=new_num_heads, student_config=student_config)
        student_blocks.append(pruned_block)
    student_model.encoder.block = nn.ModuleList(student_blocks)
    student_model.encoder.final_layer_norm = teacher_model.encoder.final_layer_norm
    return student_model

class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.loss_history = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.loss_history.append(logs["loss"])
        return control

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs_student = model(**inputs)
        device = outputs_student.last_hidden_state.device
        teacher_inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
        teacher_hidden = teacher_outputs["last_hidden_state"]
        student_logits = outputs_student.last_hidden_state / self.temperature
        teacher_logits = teacher_hidden / self.temperature
        kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction="batchmean") * (self.temperature ** 2)
        mse_loss = F.mse_loss(outputs_student.last_hidden_state, teacher_hidden)
        loss = self.alpha * mse_loss + (1 - self.alpha) * kl_loss
        return (loss, outputs_student) if return_outputs else loss

def flatten_dataset(ds):
    new_examples = []
    for example in tqdm(ds, desc="Flattening examples"):
        paragraphs = re.split(r"\n\s*\n", example["text"])
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        new_examples.extend([{"text": "categorize_topic: " + p} for p in paragraphs])
    return Dataset.from_list(new_examples)

def load_training_dataset(tokenizer, limit=50000):
    data_files = {
        "train": [
            "data/train-00000-of-00083-5a836a36820bbc21.parquet",
            "data/train-00001-of-00083-6a059492052de562.parquet",
            "data/train-00002-of-00083-6ab99ef2eda1556f.parquet",
            "data/train-00003-of-00083-fc34df8e6a0b97a4.parquet"
        ]
    }
    ds = load_dataset("cointegrated/taiga_stripped_proza", data_files=data_files, ignore_verifications=True)["train"]
    ds = flatten_dataset(ds)
    if len(ds) > limit:
        ds = ds.select(range(limit))
    def preprocess(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
    ds = ds.map(preprocess, batched=True, desc="Tokenizing dataset")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds

@record
def run_distillation(teacher_model, student_model, tokenizer, train_dataset, output_dir, num_train_epochs=1, batch_size=32, learning_rate=2e-5):
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1
    device = torch.device(f"cuda:{local_rank}") if local_rank != -1 else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    torch.cuda.set_device(device)
    if local_rank != -1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval()
    if torch.cuda.device_count() > 1 and local_rank != -1:
        teacher_model = DDP(teacher_model, device_ids=[local_rank])
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="no",
        logging_steps=10,
        save_steps=500,
        dataloader_num_workers=0,
        warmup_ratio=0.1,
        report_to=["tensorboard"],
        remove_unused_columns=False
    )
    loss_callback = LossHistoryCallback()
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        temperature=2.0,
        alpha=0.5,
        callbacks=[loss_callback]
    )
    trainer.train()
    student_path = os.path.join(output_dir, "student")
    student_model.save_pretrained(student_path)
    tokenizer.save_pretrained(student_path)
    return loss_callback.loss_history, student_path

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
    pass
