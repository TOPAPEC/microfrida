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
import logging
from transformers import T5EncoderModel, T5Config, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Attention
from datasets import load_dataset, Dataset
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

# Import evaluation function from evaluate script
from evaluate import evaluate_model

# Set up logging
root_logger = logging.getLogger()
if root_logger.handlers:
    root_logger.handlers = []
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    student_block.layer[0].SelfAttention = prune_attention_module(teacher_block.layer[0].SelfAttention, new_num_heads, student_config)
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
        pruned_block = prune_t5_block(teacher_block, new_num_heads, student_config)
        student_blocks.append(pruned_block)
    student_model.encoder.block = nn.ModuleList(student_blocks)
    student_model.encoder.final_layer_norm = teacher_model.encoder.final_layer_norm
    return student_model

def load_training_dataset(tokenizer, limit=50000, cache_file="tokenized_cache.parquet", local_rank=-1):
    if local_rank != -1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    if local_rank <= 0:
        if os.path.exists(cache_file):
            logger.info(f"Rank {local_rank}: Loading tokenized dataset from cache...")
        else:
            logger.info(f"Rank {local_rank}: Tokenized cache not found. Loading and tokenizing raw dataset...")
            # data_files = {
            #     "train": [
            #         "data/train-00000-of-00083-5a836a36820bbc21.parquet",
            #         "data/train-00001-of-00083-6a059492052de562.parquet",
            #         "data/train-00002-of-00083-6ab99ef2eda1556f.parquet",
            #         "data/train-00003-of-00083-fc34df8e6a0b97a4.parquet",
            #         "data/train-00004-of-00083-a2a0fa5d28e7d578.parquet",
            #         "data/train-00005-of-00083-806fed410a43fb46.parquet",
            #         "data/train-00006-of-00083-d6b9c39127b6005d.parquet",
            #         "data/train-00007-of-00083-206cf23f973bc8bf.parquet",
            #         "data/train-00008-of-00083-7798c30b513dd0ef.parquet",
            #         "data/train-00009-of-00083-c787d06e48840b95.parquet", 
            #         "data/train-00010-of-00083-66b33c44bfe0c7b5.parquet",
            #         "data/train-00011-of-00083-78892dcef5e8b81a.parquet",
            #         "data/train-00012-of-00083-87b563f3f9bce3cf.parquet",
            #     ]
            # }
            ds = load_dataset("cointegrated/taiga_stripped_proza", 
                            #   data_files=data_files, 
                              ignore_verifications=True,
                              cache_dir="./hf_datasets", num_proc=8)["train"]

            def flatten(example):
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", example["text"]) if p.strip()]
                return {"text": "categorize_topic: " + " ".join(paragraphs)}

            ds = ds.map(flatten, batched=False, num_proc=16)
            if len(ds) > limit:
                ds = ds.select(range(limit))

            def preprocess(examples):
                texts = examples["text"]
                return tokenizer(texts, max_length=512, padding="max_length", truncation=True)

            ds = ds.map(preprocess, batched=True, num_proc=16, desc="Tokenizing dataset")
            ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
            logger.info(f"Rank {local_rank}: Saving tokenized dataset to cache...")
            ds.to_parquet(cache_file)
    if dist.is_initialized():
        logger.info(f"Stumbled into barrier {local_rank} rank")
        dist.barrier()
    logger.info(f"Rank {local_rank}: Loading tokenized dataset from cache...")
    ds = load_dataset("parquet", data_files={"train": cache_file})["train"]
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds

def distillation_loss(student_hidden, teacher_hidden, temperature=2.0, alpha=0.5):
    student_logits = student_hidden / temperature
    teacher_logits = teacher_hidden / temperature
    kl_loss = F.kl_div(torch.log_softmax(student_logits, dim=-1),
                       torch.softmax(teacher_logits, dim=-1),
                       reduction="batchmean") * (temperature ** 2)
    mse_loss = F.mse_loss(student_hidden, teacher_hidden)
    return alpha * mse_loss + (1 - alpha) * kl_loss

@record
def run_distillation(teacher_model, student_model, tokenizer, train_dataset, output_dir,
                     num_train_epochs=1, batch_size=32, learning_rate=2e-5, epoch_to_max_length=None,
                     default_token_len=512):
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1
    if local_rank == 0:
        wandb.init(
            entity="topapec-none",
            project="microfrida",
            config={
                "learning_rate": 1e-4,
                "architecture": "t5-encoder",
                "dataset": "RussianProzaTaiga",
                "epochs": 5,
            }
        )
    device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.cuda.set_device(device)
    if local_rank != -1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval()
    if local_rank != -1 and torch.cuda.device_count() > 1:
        teacher_model = DDP(teacher_model, device_ids=[local_rank])
    sampler = DistributedSampler(train_dataset) if local_rank != -1 else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    temperature = 2.0
    alpha = 0.5
    global_step = 0
    student_model.train()
    total_loss = 0.0
    for epoch in range(num_train_epochs):
        max_tokens = epoch_to_max_length.get(epoch, default_token_len) if epoch_to_max_length is not None else default_token_len
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", disable=(local_rank != 0)):
            if isinstance(batch, list):
                batch = {k: torch.stack([item[k] for item in batch]) for k in batch[0].keys()}
            batch = {k: (torch.tensor(v) if isinstance(v, list) else v).to(device) for k, v in batch.items()}
            if 'input_ids' in batch:
                if batch['input_ids'].size(1) > max_tokens:
                    batch['input_ids'] = batch['input_ids'][:, :max_tokens]
                    if 'attention_mask' in batch:
                        batch['attention_mask'] = batch['attention_mask'][:, :max_tokens]
            with torch.no_grad():
                t_out = teacher_model(**batch) if not isinstance(teacher_model, DDP) else teacher_model.module(**batch)
            teacher_hidden = t_out.last_hidden_state
            s_out = student_model(**batch)
            student_hidden = s_out.last_hidden_state
            loss = distillation_loss(student_hidden, teacher_hidden, temperature=temperature, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            total_loss += loss.item()
            if local_rank == 0 and global_step % 10 == 0:
                logger.info(f"Epoch {epoch} Step {global_step}: Loss = {loss.item()}")
                wandb.log({"loss": loss.item(), "avg_loss": total_loss / global_step})
    logger.info(f"I AM {local_rank} gpu!")
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        student_path = os.path.join(output_dir, "student")
        student_model.eval()
        student_model.cpu()
        teacher_model.cpu()
        student_model.save_pretrained(student_path)
        tokenizer.save_pretrained(student_path)
        logger.info("Saved distilled student model and tokenizer.")
        return global_step, student_path
    else:
        return global_step, None

@record
def main():
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1
    logger.info(f"Training on local rank: {local_rank}")
    device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.cuda.set_device(device)
    if local_rank != -1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    teacher_model = T5EncoderModel.from_pretrained("ai-forever/FRIDA")
    teacher_model.to(device)
    teacher_model.eval()
    teacher_tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
    train_dataset = load_training_dataset(teacher_tokenizer, limit=10000)
    base_teacher = teacher_model.module if hasattr(teacher_model, "module") else teacher_model
    student_model = build_student_model(base_teacher, new_num_heads=12, blocks_to_keep_indices=[0, 23])
    student_model.to(device)
    output_dir = "output"
    steps, student_path = run_distillation(teacher_model, student_model, teacher_tokenizer,
                                           train_dataset, output_dir, num_train_epochs=1,
                                           batch_size=32, learning_rate=2e-5)
    
    # After training, evaluate the distilled student model using MTEB benchmark.
    if local_rank == 0 and student_path is not None:
        logger.info("Starting evaluation of the distilled model using MTEB benchmark...")
        results, stats = evaluate_model(student_path, tasks=["GeoreviewClusteringP2P"], eval_output_folder="eval_results")
        # Assuming that results is a dictionary containing "main_score"
        main_score = results["scores"]["test"][0].get("main_score")
        if main_score is not None:
            wandb.log({"mteb_main_score": main_score})
            logger.info(f"Logged MTEB main_score: {main_score} to wandb")
        else:
            logger.warning("MTEB main_score not found in evaluation results.")
    
    return steps, student_path

if __name__ == "__main__":
    main()
