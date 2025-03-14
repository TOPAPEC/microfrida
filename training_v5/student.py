import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

class FridaDistillationModel(nn.Module):
    def __init__(self, hidden_size=368, num_heads=8, num_layers=3, device=None):
        super(FridaDistillationModel, self).__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.teacher_model = T5EncoderModel.from_pretrained("ai-forever/FRIDA").to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.teacher_embedding_dim = self.teacher_model.shared.weight.shape[1]
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_size // num_heads
        
        self.embedding_transform = nn.Linear(self.teacher_embedding_dim, hidden_size).to(self.device)
        
        self.teacher_embeddings = nn.Parameter(
            self.teacher_model.shared.weight.clone(), 
            requires_grad=False
        ).to(self.device)
        
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, hidden_size * 4).to(self.device)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_size).to(self.device)
        self.to(self.device)
        
    def get_embedding(self, input_ids):
        # Use teacher embeddings and apply transformation
        embeddings = F.embedding(input_ids, self.teacher_embeddings)
        return self.embedding_transform(embeddings)
    
    def forward(self, input_ids, attention_mask=None):
        # attention_mask = attention_mask.to(self.device)
        # input_ids = input_ids.to(self.device)
        hidden_states = self.get_embedding(input_ids)
        
        # Process through transformer layers
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=self.device).to(self.device)
            
        # Convert attention mask for transformer
        extended_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        
        # Process through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_mask)
            
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states
    
    def get_qkv_matrices(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        """Get QKV matrices for distillation loss calculation"""
        # Student QKV
        hidden_states = self.forward(input_ids, attention_mask)
        last_layer = self.layers[-1]
        student_q = last_layer.self_attention.q(hidden_states)
        student_k = last_layer.self_attention.k(hidden_states)
        student_v = last_layer.self_attention.v(hidden_states)
        
        # Reshape for multi-head
        batch_size, seq_len, _ = student_q.shape
        student_q = student_q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        student_k = student_k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        student_v = student_v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Teacher QKV
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_hidden = teacher_outputs.last_hidden_state
            
            teacher_encoder = self.teacher_model.encoder
            teacher_last_layer = teacher_encoder.block[-1]
            teacher_attention = teacher_last_layer.layer[0].SelfAttention
            
            teacher_q = teacher_attention.q(teacher_hidden)
            teacher_k = teacher_attention.k(teacher_hidden)
            teacher_v = teacher_attention.v(teacher_hidden)
            
            # Reshape for multi-head
            teacher_head_dim = teacher_attention.key_value_proj_dim
            teacher_heads = teacher_attention.n_heads
            teacher_q = teacher_q.view(batch_size, seq_len, teacher_heads, teacher_head_dim).transpose(1, 2)
            teacher_k = teacher_k.view(batch_size, seq_len, teacher_heads, teacher_head_dim).transpose(1, 2)
            teacher_v = teacher_v.view(batch_size, seq_len, teacher_heads, teacher_head_dim).transpose(1, 2)
        
        return {
            'student': {'q': student_q, 'k': student_k, 'v': student_v},
            'teacher': {'q': teacher_q, 'k': teacher_k, 'v': teacher_v}
        }
    
    def compute_distillation_loss(self, input_ids, attention_mask=None):
        """Compute MiniLMv2 distillation loss with relation heads projection"""
        qkv_dict = self.get_qkv_matrices(input_ids, attention_mask)
        
        student_q, student_k, student_v = qkv_dict['student'].values()
        teacher_q, teacher_k, teacher_v = qkv_dict['teacher'].values()
        
        # Get shapes and determine relation heads
        batch_size, student_heads, seq_len, student_head_dim = student_q.shape
        _, teacher_heads, _, teacher_head_dim = teacher_q.shape
        
        # Set number of relation heads (can be different from both teacher and student)
        # Using a fixed number like 12 or 24 as in the paper
        num_relation_heads = 8
        
        # Project to relation heads through reshape operations
        # Student relation projections
        student_q_for_relation = student_q.transpose(1, 2).reshape(batch_size, seq_len, -1)
        student_k_for_relation = student_k.transpose(1, 2).reshape(batch_size, seq_len, -1)
        student_v_for_relation = student_v.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        # Split to relation heads
        student_rq = student_q_for_relation.view(batch_size, seq_len, num_relation_heads, -1).transpose(1, 2)
        student_rk = student_k_for_relation.view(batch_size, seq_len, num_relation_heads, -1).transpose(1, 2)
        student_rv = student_v_for_relation.view(batch_size, seq_len, num_relation_heads, -1).transpose(1, 2)
        
        # Teacher relation projections
        teacher_q_for_relation = teacher_q.transpose(1, 2).reshape(batch_size, seq_len, -1)
        teacher_k_for_relation = teacher_k.transpose(1, 2).reshape(batch_size, seq_len, -1)
        teacher_v_for_relation = teacher_v.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        # Split to relation heads
        teacher_rq = teacher_q_for_relation.view(batch_size, seq_len, num_relation_heads, -1).transpose(1, 2)
        teacher_rk = teacher_k_for_relation.view(batch_size, seq_len, num_relation_heads, -1).transpose(1, 2)
        teacher_rv = teacher_v_for_relation.view(batch_size, seq_len, num_relation_heads, -1).transpose(1, 2)
        
        # Compute relations with aligned dimensions
        student_relation_dim = student_rq.size(-1)
        teacher_relation_dim = teacher_rq.size(-1)
        
        # Student relations
        student_q_q = torch.matmul(student_rq, student_rq.transpose(-1, -2)) / torch.sqrt(torch.tensor(student_relation_dim, dtype=student_rq.dtype))
        student_k_k = torch.matmul(student_rk, student_rk.transpose(-1, -2)) / torch.sqrt(torch.tensor(student_relation_dim, dtype=student_rk.dtype))
        student_v_v = torch.matmul(student_rv, student_rv.transpose(-1, -2)) / torch.sqrt(torch.tensor(student_relation_dim, dtype=student_rv.dtype))
        
        # Teacher relations
        teacher_q_q = torch.matmul(teacher_rq, teacher_rq.transpose(-1, -2)) / torch.sqrt(torch.tensor(teacher_relation_dim, dtype=teacher_rq.dtype))
        teacher_k_k = torch.matmul(teacher_rk, teacher_rk.transpose(-1, -2)) / torch.sqrt(torch.tensor(teacher_relation_dim, dtype=teacher_rk.dtype))
        teacher_v_v = torch.matmul(teacher_rv, teacher_rv.transpose(-1, -2)) / torch.sqrt(torch.tensor(teacher_relation_dim, dtype=teacher_rv.dtype))
        
        # Apply softmax to get proper probability distributions
        student_q_q = F.softmax(student_q_q, dim=-1)
        student_k_k = F.softmax(student_k_k, dim=-1)
        student_v_v = F.softmax(student_v_v, dim=-1)
        
        teacher_q_q = F.softmax(teacher_q_q, dim=-1)
        teacher_k_k = F.softmax(teacher_k_k, dim=-1)
        teacher_v_v = F.softmax(teacher_v_v, dim=-1)
        
        # KL divergence for distillation
        with torch.cuda.amp.autocast(enabled=False):
            loss_q_q = F.kl_div(torch.log(student_q_q + 1e-10), teacher_q_q, reduction='batchmean', log_target=False)
            loss_k_k = F.kl_div(torch.log(student_k_k + 1e-10), teacher_k_k, reduction='batchmean', log_target=False)
            loss_v_v = F.kl_div(torch.log(student_v_v + 1e-10), teacher_v_v, reduction='batchmean', log_target=False)
        
        total_loss = loss_q_q + loss_k_k + loss_v_v
        return total_loss


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        residual = x
        x = self.layer_norm(x)
        
        # Linear projections
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention calculation
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Apply attention
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.o(context)
        
        return output + residual

class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_dim):
        super(FeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_dim):
        super(TransformerLayer, self).__init__()
        self.self_attention = SelfAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, ff_dim)
        
    def forward(self, x, mask=None):
        x = self.self_attention(x, mask)
        x = self.feed_forward(x)
        return x
