import re
from collections import Counter
import json

class CustomTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    
    def clean_text(self, text):
        # Basic text cleaning
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = text.strip()
        return text
    
    def tokenize(self, text):
        # Split text into tokens
        return text.split()
    
    def build_vocab(self, texts):
        # Build vocabulary from a list of texts
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(self.clean_text(text))
            counter.update(tokens)
        
        # Most common tokens
        most_common_tokens = [token for token, _ in counter.most_common(self.vocab_size - len(self.special_tokens))]
        
        # Create vocabulary dictionary
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens + most_common_tokens)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
    
    def encode(self, text):
        # Encode text into a list of token ids
        tokens = self.tokenize(self.clean_text(text))
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
    
    def decode(self, token_ids):
        # Decode a list of token ids back into text
        return ' '.join([self.inverse_vocab.get(token_id, '<UNK>') for token_id in token_ids])
    
    def save_vocab(self, filepath):
        # Save vocabulary to a file
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f)
    
    def load_vocab(self, filepath):
        # Load vocabulary from a file
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
    
    def train(self, dataset_path):
        # Train tokenizer on a dataset
        with open(dataset_path, 'r') as f:
            texts = f.readlines()
        self.build_vocab(texts)

class BasicTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def encode(self, text):
        return [self.vocab.get(ch, -1) for ch in text]
    
    def decode(self, token_ids):
        return ''.join([self.inverse_vocab.get(token_id, '') for token_id in token_ids])
    
    def hi(self, text):
        return text

    def build_vocab(self, chars):
        self.vocab = {ch:i for i,ch in enumerate(chars)}
        self.inverse_vocab = {i:ch for i,ch in enumerate(chars)}

        


# Example usage
if __name__ == "__main__":
    # tokenizer = CustomTokenizer(vocab_size=10000)
    # tokenizer.train('path_to_dataset.txt')
    # encoded_text = tokenizer.encode("Hello, world!")
    # decoded_text = tokenizer.decode(encoded_text)
    # print(f"Encoded: {encoded_text}")
    # print(f"Decoded: {decoded_text}")
    # tokenizer.save_vocab('vocab.json')
    # tokenizer.load_vocab('vocab.json')
    tokenizer = BasicTokenizer()
    print(tokenizer.hi('h'))
