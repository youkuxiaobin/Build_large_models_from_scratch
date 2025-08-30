import torch
import math
import tiktoken

class MutiHeadAttentionV1(torch.nn.Module):
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
        
        
        
class LayerNorm(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(cfg["n_embd"]))
        self.shift = torch.nn.Parameter(torch.zeros(cfg["n_embd"]))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm_x = (x - mean)/torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.eps
    
class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1+torch.tanh(torch.sqrt(torch.tensor(2.0)/torch.pi)*(x+0.044715*torch.pow(x,3))))
    
class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(cfg["n_embd"], 4*cfg["n_embd"]),
            GELU(),
            torch.nn.Linear(4*cfg["n_embd"], cfg["n_embd"]))
    def forward(self, x):
        return self.net(x)
class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MutiHeadAttentionV1(d_in = cfg["n_embd"],
                                       d_out = cfg["n_embd"],
                                       num_head = cfg["n_head"],
                                       context_length = cfg["context_length"],
                                       dropout=cfg["dropout_rate"],
                                       bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg)
        self.norm2 = LayerNorm(cfg)
        self.dropout = torch.nn.Dropout(cfg["dropout_rate"])
    def forward(self, x):
     
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = shortcut + x
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = shortcut + x
        return x
        
class DumpGPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["n_embd"])
        self.dropout = torch.nn.Dropout(cfg["dropout_rate"])

        self.trf_blocks = torch.nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layer"])])
        self.norm_block = LayerNorm(cfg)
        self.out_head = torch.nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias = False)
        

    def forward(self, x):
        batch_size, token_num = x.shape

        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(token_num))
        inputs = token_emb + pos_emb

        #dropout
        inputs = self.dropout(inputs)
        #transformer block的处理
        attention_scores = self.trf_blocks(inputs)
        #norm layer
        normed_output = self.norm_block(attention_scores)
        #output layer
        logits = self.out_head(normed_output)
        return logits

if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    txt1 = "Every effort move you"
    txt2 = "Every day holds a"

    batch = []
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    print(batch)
    #[tensor([6109, 3626, 1445,  345]), tensor([6109, 1110, 6622,  257])]
    batch = torch.stack(batch, dim=0)
    print(batch)
    #tensor([[6109, 3626, 1445,  345],
    #    [6109, 1110, 6622,  257]])
    torch.manual_seed(123)
    model = DumpGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print(logits.shape)
    #torch.Size([2, 4, 50257])
    print(logits)   