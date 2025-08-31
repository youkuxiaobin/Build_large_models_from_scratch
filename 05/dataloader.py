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

def create_dataloader_v1(txt, max_length=4, stride=1, shuffle=False, batch_size=1, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = SimpleTokenizerV1(txt, tokenizer, max_length=max_length, stride=stride)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers = 0
    )
    return dataloader

    
