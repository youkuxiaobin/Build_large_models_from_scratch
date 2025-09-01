import tiktoken
import torch
from dataloader import create_dataloader_v1
from model import DumpGPTModel
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0
    if len(data_loader) == 0 :
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else :
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, 
                       optimizer, device, num_epochs, 
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % eval_freq == 0 :
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch +1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f},"
                      f"Val loss {val_loss:.3f}")
                
            generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen
def evaluate_model(model, train_loader, valid_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(valid_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, val_loss


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range (max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, idx=encoded, max_new_tokens=50, context_size=context_size)
    
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, axl = plt.subplots(figsize=(5,3))
    axl.plot(epochs_seen, train_losses, label="Training loss")
    axl.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    axl.set_xlabel("Epochs")
    axl.set_ylabel("Loss")
    
    axl.legend(loc="upper right")
    axl.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = axl.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
    plt.savefig("loss.png")
    
if __name__ == "__main__":
    
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    valid_data = text_data[split_idx:]    
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }
    
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data, max_length = GPT_CONFIG_124M["context_length"],
        stride = GPT_CONFIG_124M["context_length"],
        shuffle=True,
        batch_size = 2,
        drop_last = True
    )
    
    
    valid_loader = create_dataloader_v1(
        valid_data,
        max_length = GPT_CONFIG_124M["context_length"],
        stride = GPT_CONFIG_124M["context_length"],
        shuffle=False,
        batch_size = 2,
        drop_last = False
    )
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    print("Valid loader") 
    for x, y in valid_loader:
        print(x.shape, y.shape)   
        
    model = DumpGPTModel(GPT_CONFIG_124M)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(valid_loader, model, device)
    
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epoches = 10
    
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader,
                                valid_loader, optimizer, device, num_epochs=num_epoches,
                                eval_freq=5, eval_iter = 5, start_context="Every effort moves you",
                                tokenizer=tokenizer 
                                )
    epochs_tensor = torch.linspace(0, num_epoches, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    
    torch.save(model.state_dict(), "model.pth")
    