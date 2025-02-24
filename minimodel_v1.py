#!/usr/bin/env python3
import copy
import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Config, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Attention

def prune_linear_layer_generic(linear_layer, top_indices, num_heads_teacher, head_dim):
    d_model = linear_layer.in_features
    # For q, k, v layers: weight is of shape (out_features, in_features).
    # We transpose to (in_features, out_features) so that we can view the weights by head.
    w = linear_layer.weight.data.t().contiguous().view(d_model, num_heads_teacher, head_dim)
    pruned_w = w[:, top_indices, :].contiguous().view(d_model, len(top_indices) * head_dim)
    pruned_w = pruned_w.t().contiguous()  # Now shape becomes (new_num_heads*head_dim, d_model)
    pruned_b = None
    if linear_layer.bias is not None:
        b = linear_layer.bias.data.view(num_heads_teacher, head_dim)
        pruned_b = b[top_indices, :].contiguous().view(len(top_indices) * head_dim)
    return pruned_w, pruned_b

def prune_linear_layer_o(o_layer, top_indices, num_heads_teacher, head_dim):
    # For o layer: weight shape is (d_model, num_heads_teacher * head_dim)
    # We transpose to (in_features, d_model) and then view as (num_heads_teacher, head_dim, d_model).
    d_model = o_layer.weight.data.size(0)
    w = o_layer.weight.data.t().contiguous().view(num_heads_teacher, head_dim, d_model)
    pruned_w = w[top_indices, :, :].contiguous().view(len(top_indices) * head_dim, d_model)
    pruned_w = pruned_w.t().contiguous()  # Expected shape: (d_model, new_num_heads*head_dim)
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
    num_heads_teacher = attn_module.n_heads  # Use correct attribute from T5Attention
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
    student_block.layer[0].SelfAttention = prune_attention_module(
        teacher_block.layer[0].SelfAttention,
        new_num_heads=new_num_heads,
        student_config=student_config
    )
    return student_block

def build_student_model(teacher_model, new_num_heads=12, blocks_to_keep_indices=[0, 1, 20, 21, 22, 23]):
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

if __name__ == "__main__":
    teacher_model_id = "ai-forever/FRIDA"
    teacher_model = T5EncoderModel.from_pretrained(teacher_model_id)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    student_model = build_student_model(
        teacher_model,
        new_num_heads=6,
        blocks_to_keep_indices=[0, 8, 20, 21, 22, 23]
    )
    print("Teacher layers:", len(teacher_model.encoder.block))
    print("Student layers:", len(student_model.encoder.block))
    print("Student self-attention heads (example):", student_model.encoder.block[0].layer[0].SelfAttention.n_heads)
    student_model.save_pretrained("./student_FRIDA")
    tokenizer.save_pretrained("./student_FRIDA")
    print("Student model and tokenizer saved.")

