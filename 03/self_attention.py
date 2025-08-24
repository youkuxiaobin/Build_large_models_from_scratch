import torch
class SelfAttentionLayerV1(torch.nn.Module):
    def __init__(self, in_dim, out_dim,dropout):
        super().__init__()
        
        self.Key = torch.nn.Linear(in_dim, out_dim)
        self.Query = torch.nn.Linear(in_dim, out_dim)
        self.Value = torch.nn.Linear(in_dim, out_dim)
        self.Dropout = torch.nn.Dropout(dropout)
        
    def forward(self, inputs):
        key = self.Key(inputs) # (6, 2)
        val = self.Value(inputs) # (6, 2)
        query = self.Query(inputs) # (6, 2)
        attention_scores =  query @ key.transpose(-2, -1) # 6, 6
        mask = torch.triu(torch.ones(attention_scores.shape[-1], attention_scores.shape[-1]),diagonal=1)
        masked_attention = attention_scores.masked_fill(mask.bool(), -torch.inf)
        print(masked_attention)
        
        attention = torch.softmax(masked_attention/ (key.size(-1) ** 0.5), dim=-1) 
        attention = self.Dropout(attention)
        
        return attention @val
    
if __name__ == "__main__":
    torch.manual_seed(123)
    sa_v2 = SelfAttentionLayerV1(3, 2, 0.5)
    inputs = torch.tensor(
    [[0.43, 0.15, 0.89 ],   #Your
     [0.55, 0.87, 0.66 ],   #journey
     [0.57, 0.85, 0.64 ],   #starts
     [0.22, 0.58, 0.33 ],   #with
     [0.77, 0.25, 0.10 ],   #one
     [0.05, 0.80, 0.55]]    #step
    )
    print(sa_v2(inputs))