import urllib.request
import torch
import tiktoken
class SimpleTokenizerV1(torch.utils.data.Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids =  []
        
        tokens_ids = tokenizer.encode(txt)
        for i in range(0, len(tokens_ids)-max_length, stride):
            self.input_ids.append(torch.tensor(tokens_ids[i: i+max_length]))
            self.target_ids.append(torch.tensor(tokens_ids[i+1: i+max_length+1]))
            
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    def __len__(self):
        return len(self.input_ids)        

def create_dataloader_v1(txt, max_length=4, stride=1, shuffle=False, batch_size=1):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = SimpleTokenizerV1(txt, tokenizer, max_length=max_length, stride=stride)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=True,
    )
    return dataloader
if __name__ == "__main__":
    file_path = "the-verdct.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    max_length = 4
    dataloader = create_dataloader_v1(raw_text, max_length=max_length, stride=1, shuffle=False)
    
    data_iter = iter(dataloader)
    inputs, target = next(data_iter)
    print(inputs, target)
    
    emb_layer = torch.nn.Embedding(num_embeddings=50257, embedding_dim=256)
    print(emb_layer(inputs).shape)
    pos_emb_layer = torch.nn.Embedding(num_embeddings=max_length, embedding_dim=256)
    pos_embed = pos_emb_layer(torch.arange(max_length))
    print(pos_embed.shape)
    
