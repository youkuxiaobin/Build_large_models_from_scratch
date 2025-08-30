import torch
import math

class MutiSelfAttentionV1(torch.nn.Module):
    def __init__(self, d_in, d_out, num_head, context_length, dropout,  bias = False):
        super().__init__()
        
        assert d_out % num_head == 0
        
        self.d_out = d_out
        self.num_head = num_head
        self.head_dim = d_out // num_head
        self.Key = torch.nn.Linear(d_in, d_out, bias=bias) 
        self.Value = torch.nn.Linear(d_in, d_out, bias=bias)
        self.Query = torch.nn.Linear(d_in, d_out, bias=bias)
        
        self.out_proj = torch.nn.Linear(d_out, d_out, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1))
    def forward(self, inputs):
        batch_size, token_size, _ = inputs.shape
        
        key = self.Key(inputs)
        value = self.Value(inputs)
        query = self.Query(inputs)
        
        key = key.view(batch_size, token_size, self.num_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, token_size, self.num_head, self.head_dim).transpose(1, 2)
        query = query.view(batch_size, token_size, self.num_head, self.head_dim).transpose(1, 2)
        
        attention_scores = (query @ key.transpose(-1, -2))/math.sqrt(self.head_dim) 
        mask = self.mask[:token_size, :token_size]   
        attention_scores.masked_fill_(mask, -torch.inf)
        
        attention_weigth = torch.softmax(attention_scores, dim=-1) 
        attention_weigth = self.dropout(attention_weigth)
        ctx = (attention_weigth @ value).transpose(1, 2).contiguous().view(batch_size, token_size, self.d_out)
        out = self.out_proj(ctx)
        return out
        
        
        
        