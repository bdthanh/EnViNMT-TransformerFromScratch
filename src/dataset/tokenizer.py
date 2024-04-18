from abc import ABC, abstractmethod
import spacy
from underthesea import word_tokenize
from dataset.vocabulary import Vocabulary, ParallelVocabulary

class BaseTokenizer(ABC):

    def __init__(self, vocab_fpath=None):
        self.vocab = Vocabulary()
        if vocab_fpath:
            self.load_vocab(vocab_fpath)

    @abstractmethod
    def tokenize(self, sent):
        pass
    
    @abstractmethod
    def detokenize(self, tokens):
        pass

    def build_vocab(self, sents, is_tokenized=False, min_freq=1, vocab_size=None):
        if not is_tokenized:
            tokenized_sents = self.tokenize(sents)
        else:
            tokenized_sents = sents
        self.vocab.add_tokens(tokenized_sents, min_freq, vocab_size)

    def save_vocab(self, vocab_fpath):
        with open(vocab_fpath, "w") as f:
            for token in self.vocab.token_to_id.keys(): 
                f.write(token + ("\n"))

    def load_vocab(self, vocab_fpath):
        if vocab_fpath is not None:
            with open(vocab_fpath, "r") as f:
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
        
    