import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
from torchtext.vocab import vocab as Vocab
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from utils import Config



class DataProcessor:
    def __init__(self,config) -> None:
        self.config = config
        self.tokenizer = WordPunctTokenizer()
        self.src_vocab = None
        self.trg_vocab = None
    def tokenize(self,text):
        return self.tokenizer.tokenize(text.rstrip().lower())
    def build_vocab(self,train_data):
        src_counter = Counter()
        trg_counter = Counter()
        for src,trg in train_data:
            src_counter.update(self.tokenize(src))
            trg_counter.update(self.tokenize(trg))
        self.src_vocab = Vocab(src_counter, min_freq = self.config.MIN_FREQ)
        self.trg_vocab = Vocab(trg_counter,min_freq = self.config.MIN_FREQ)
        
        special_tokens = [
            self.config.UNK_TOKEN,
            self.config.PAD_TOKEN,
            self.config.SOS_TOKEN,
            self.config.EOS_TOKEN
        ]
        
        for vocab in [self.src_vocab,self.trg_vocab]:
            if self.confg.UNK_TOKEN not in vocab:
                vocab.insert_token(self.config.UNK_TOKEN, index=0)
                vocab.set_default_index(0)
                
            for token in special_tokens[1:]:
                if token not in vocab:
                    vocab.append_token(token)
                    
        print(f'Source Vocablary Size : {len(self.src_vocab)}')
        print(f'Target Vocablary Size : {len(self.trg_vocab)}')
        
        return self.src_vocab, self.trg_vocab
    
    def encode(self,sent,vocab,reverse=False):
        tokenized = [self.config.SOS_TOKEN] + self.tokenize(sent) + [self.config.EOS_TOKEN]
        encoded = vocab.lookup_indices(tokenized)
        if reverse:
            encoded = encoded[::-1]
        return encoded
    def collate_batch(self,batch):
        src_list, trg_list = [], []
        for src,trg in batch:
            src_encoded = self.encode(src,self.src_vocab, reverse=True)
            trg_encoded = self.encode(trg,self.trg_vocab,reverse=False)
            
            src_list.append(torch.tensor(src_encoded))
            trg_list.append(torch.tensor(trg_encoded))
        src_padded = pad_sequence(src_list,padding_value = self.src_vocab[self.config.PAD_TOKEN])
        trg_padded = pad_sequence(trg_list,padding_value = self.trg_vocab[self.config.PAD_TOKEN])
        return src_padded, trg_padded
    def get_dataloaders(self):
        train_data = list(Multi30k(split='train',language_pair = (self.config.SRC_LANG,self.config.TRG_LANG)))
        val_data = list(Multi30k(split='test', language_pair = (self.config.SRC_LANG,self.config.TRG_LANG)))  
        
        print(f"Training examples : {len(train_data)}")          
        print(f"Test examples : {len(val_data)}")
        
        self.build_vocab(train_data)
        train_loader = DataLoader(
            train_data,
            batch_size = self.config.BATCH_SIZE,
            collate_fn = self.collate_batch,
            shuffle = True
            )    
        val_loader = DataLoader(
            val_data,
            batch_size = self.config.BATCH_SIZE,
            collate_fn = self.collate_batch,
            shuffle = False
        )      
        
        return train_loader, val_loader, train_data, val_data