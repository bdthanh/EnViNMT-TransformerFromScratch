import torch
from typing import List
from collections import Counter
from itertools import chain 

class Vocabulary:
    def __init__(self):
        self.unk_token = '<unk>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.token_to_id = dict()
        self.token_to_id[self.unk_token] = 0
        self.token_to_id[self.sos_token] = 1
        self.token_to_id[self.eos_token] = 2
        self.token_to_id[self.pad_token] = 3
        self.unk_id = self.token_to_id[self.unk_token]
        self.sos_id = self.token_to_id[self.sos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.pad_id = self.token_to_id[self.pad_token]
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
    def __len__(self):
        return len(self.token_to_id)
      
    def __getitem__(self, token):
        return self.token_to_id.get(token, self.unk_id)
    
    def add(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
            self.id_to_token[self.token_to_id[token]] = token
            
    def sentence_to_tensor(self, tokenized_sent: List[str]):
        return torch.tensor(tokenized_sent, dtype=torch.int64)
      
    def corpus_to_tensors(self, tokenized_corpus):
        return [self.sentence_to_tensor(sentence) for sentence in tokenized_corpus]
    
    def tensor_to_sentence(self, tensor):
        return list(map(lambda id: self.id_to_token[id.item()], tensor))
      
    def tensors_to_corpus(self, tensors):
        return [self.tensor_to_sentence(tensor) for tensor in tensors]
          
    def add_tokens(self, tokenized_corpus, min_freq=1, vocab_size=None):
        token_freq = Counter(chain(*tokenized_corpus))
        freq_filter_tokens = [token for token in token_freq if token_freq[token] >= min_freq]
        if vocab_size is not None: # limit the vocab size if specify
            freq_filter_tokens = sorted(freq_filter_tokens, key=lambda token: token_freq[token], reverse=True)[:vocab_size]
        for token in freq_filter_tokens:
            self.add(token)
        
if __name__ == "__main__":
    vocab = Vocabulary()
    tokenized_corpus = [['I', 'am', 'a', 'student'], ['you', 'are', 'a', 'teacher']]
    vocab.add_tokens(tokenized_corpus=tokenized_corpus, min_freq=1)
    print(f'Token to id: {vocab.token_to_id}')
    print(f'ID to token: {vocab.id_to_token}')
    