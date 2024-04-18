from abc import ABC, abstractmethod
import os
import spacy
from underthesea import word_tokenize
from vocabulary import Vocabulary

class BaseTokenizer(ABC):

    def __init__(self, vocab_fpath=None):
        self.vocab = Vocabulary()
        if vocab_fpath:
            self.load_vocab(vocab_fpath)

    @abstractmethod
    def tokenize(self, sentence):
        pass
    
    @abstractmethod
    def detokenize(self, tokens):
        pass

    def build_vocab(self, corpus, is_tokenized=False, min_freq=1, vocab_size=None):
        if not is_tokenized:
            tokenized_corpus = [self.tokenize(sentence) for sentence in corpus]
        else:
            tokenized_corpus = corpus
        self.vocab.add_tokens(tokenized_corpus, min_freq, vocab_size)
        return tokenized_corpus

    def save_vocab(self, vocab_fpath):
        with open(vocab_fpath, "w", encoding='utf-8') as f:
            for token in self.vocab.token_to_id.keys(): 
                f.write(token + ("\n"))

    def load_vocab(self, vocab_fpath):
        if vocab_fpath is not None:
            with open(vocab_fpath, "r", encoding='utf-8') as f:
                for token in f: 
                    self.vocab.add(token.rstrip("\n"))
        

class ViTokenizer(BaseTokenizer):
    def __init__(self, vocab_fpath=None):
        super().__init__(vocab_fpath)

    def tokenize(self, sentence):
        if len(sentence) == 0:
            return list()
        else:
            return word_tokenize(sentence)
        
    def detokenize(self, tokens):
        return " ".join(tokens)


class EnTokenizer(BaseTokenizer):
    def __init__(self, vocab_fpath=None):
        super().__init__(vocab_fpath)
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize(self, sentence):
        return [tok.text for tok in self.spacy_en.tokenizer(sentence)]
    
    def detokenize(self, tokens):
        return " ".join(tokens)
      
if __name__ == "__main__":
    vi_sentence = "Nó là một trò chơi ghép hình vẫn đang được xếp"
    en_sentence = "It is a jigsaw puzzle still being put together"
    vi_tokenizer = ViTokenizer()
    en_tokenizer = EnTokenizer()
    vi_tokenized = vi_tokenizer.tokenize(vi_sentence)
    en_tokenized = en_tokenizer.tokenize(en_sentence)
    print(vi_tokenized)
    print(en_tokenized)
    vi_tokenizer.build_vocab([vi_tokenized], is_tokenized=True)
    en_tokenizer.build_vocab([en_tokenized], is_tokenized=True)
    vi_tokenizer.save_vocab(vocab_fpath='vi.txt')
    en_tokenizer.save_vocab(vocab_fpath='en.txt')
                   