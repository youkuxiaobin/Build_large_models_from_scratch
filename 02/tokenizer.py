import urllib.request
import re

class SimpleTokenizerV1:
    def __init__(self):
        url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
        file_path = "the-verdict.txt"
        urllib.request.urlretrieve(url, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in result if item.strip()]

        all_words = sorted(set(preprocessed))
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        self.str2id = {token:integer for integer, token in enumerate(all_words)}
        self.id2str = {integer:token for integer, token in enumerate(all_words)}
    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() if item in self.str2id else "<|unk|>" for item in preprocessed if item.strip()]
        return [self.str2id[s] for s in preprocessed]
    def decode(self, ids):
        return " ".join([self.id2str[i] for i in ids])
    
text = """It's the last he painted, you know, Mrs Gisburn said with pardonable pride."""
tokenizer = SimpleTokenizerV1()
print(tokenizer.encode(text))

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace"
text = "<|endoftext|> ". join((text1, text2))
print(tokenizer.encode(text))    
print(tokenizer.decode(tokenizer.encode(text)))