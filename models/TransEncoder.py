import torch
import torch.nn as nn


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super(MultiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size, seq_len, d_model = q.size()
        Q = self.W_q(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(output)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size, seq_len, d_model = q.size()
        Q = self.W_q(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, attention_type="multi_head"):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        if attention_type == "multi_head":
            self.attention = MultiHeadAttention(d_model, num_heads)
        elif attention_type == "multi_query":
            self.attention = MultiQueryAttention(d_model, num_heads)
        else:
            raise ValueError("Unrecognized attention type. Supported types are'multi_head' and'multi_query'.")

    def forward(self, x):
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, attention_type="multi_head"):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, attention_type)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# # 示例使用
# batch_size = 16
# seq_length = 32
# d_model = 512
# num_layers = 4
# num_heads = 8
#
# # 使用多头注意力
# input_tensor = torch.randn(batch_size, seq_length, d_model)
# encoder_multi_head = TransformerEncoder(d_model, num_layers, num_heads, attention_type="multi_head")
# output_multi_head = encoder_multi_head(input_tensor)
# print("多头注意力输出形状:", output_multi_head.shape)
#
# # 使用多查询注意力
# encoder_multi_query = TransformerEncoder(d_model, num_layers, num_heads, attention_type="multi_query")
# output_multi_query = encoder_multi_query(input_tensor)
# print("多查询注意力输出形状:", output_multi_query.shape)