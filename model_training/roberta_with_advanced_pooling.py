import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaModel, AutoModel, AutoConfig


# Written with the help of Claude.ai

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output, attention_mask=None):
        # sequence_output: (batch_size, seq_len, hidden_size)
        attention_weights = self.attention(sequence_output)  # (batch_size, seq_len, 1)
        attention_weights = attention_weights.squeeze(-1)  # (batch_size, seq_len)

        if attention_mask is not None:
            # Mask out padded positions
            attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)

        attention_weights = torch.softmax(attention_weights, dim=-1)  # (batch_size, seq_len)
        attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)

        # Weighted sum
        pooled_output = torch.sum(sequence_output * attention_weights, dim=1)  # (batch_size, hidden_size)
        return pooled_output


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # Learnable query vector for pooling
        self.pooling_query = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, sequence_output, attention_mask=None):
        batch_size, seq_len, hidden_size = sequence_output.shape

        # Expand pooling query for batch
        pooling_query = self.pooling_query.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)

        # Project to Q, K, V
        Q = self.query(pooling_query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(sequence_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(sequence_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            # Expand mask for multi-head attention
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, 1, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, hidden_size)

        # Output projection
        pooled_output = self.output_proj(context).squeeze(1)  # (batch_size, hidden_size)
        return pooled_output


def mean_pooling(sequence_output, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
    sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    pooled_output = sum_embeddings / sum_mask
    return pooled_output


def max_pooling(sequence_output, attention_mask):
    pooled_output = torch.max(sequence_output, dim=1)[0]
    return pooled_output


def cls_pooling(sequence_output, attention_mask=None):
    pooled_output = sequence_output[:, 0]
    return pooled_output


class RobertaWithAdvancedPooling(nn.Module):
    def __init__(self, model_name, num_labels, pooling_method='mean', dropout=0.1, num_attention_heads=8, hidden_dim=None, initialize_roberta=True):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, 'dtype'):
            delattr(config, 'dtype')

        self.roberta = RobertaModel.from_pretrained(model_name) if initialize_roberta else RobertaModel._from_config(config)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.pooling_method = pooling_method
        self.num_attention_heads = num_attention_heads

        hidden_size = self.roberta.config.hidden_size
        if hidden_dim:
            # Add an intermediate layer
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels)
            )
        else:
            # Direct mapping to labels
            self.classifier = nn.Linear(hidden_size, num_labels)

        if pooling_method == 'attention':
            self.pooling = AttentionPooling(self.roberta.config.hidden_size)
        elif pooling_method == 'multihead_attention':
            self.pooling = MultiHeadAttentionPooling(self.roberta.config.hidden_size, num_attention_heads)
        elif pooling_method == 'mean':
            self.pooling = mean_pooling
        elif pooling_method == 'max':
            self.pooling = max_pooling
        else:  # cls pooling
            self.pooling = cls_pooling

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        pooled_output = self.pooling(sequence_output, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return {"logits": logits}

