import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def   __init__(self,n_tokens,emb_dim,hid_dim, n_layers,dropout,pad_idx) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(n_tokens,emb_dim,padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim,hid_dim,n_layers,dropout=dropout)
        
    def forward(self,src):
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        _, hidden =  self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self,n_tokens,emb_dim,hid_dim, n_layers,dropout,pad_idx) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_tokens, emb_dim,padding_idx = pad_idx)
        self.rnn = nn.LSTM(emb_dim,hid_dim,n_layers,dropout = dropout)
        self.out = nn.Linear(hid_dim, n_tokens)
        
    def forward(self,input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input),
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded,hidden)
        preds = self.out(output.squeeze(0))
        return preds, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hid_dim == decoder.hid_dim , "Encoder and Decoder must have the same hidden dim"
        assert encoder.n_layers == decoder.n_layers , "Encoder and Decoder must have the same hidden n_layers"
        
    def forward(self,src,trg,teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.n_tokens
        preds = []
        hidden = self.encoder(src)
        input = trg[0,:]
        for i in range(1,trg_len):
            pred,hidden = self.decoder(input, hidden)
            preds.append(pred)
            teacher_force = random.random() < teacher_forcing_ratio
            _, top_pred = pred.max(dim=1)
            input = trg[i,:] if teacher_force else top_pred
        return torch.stack(preds)
    
    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform(param,-0.08,0.08)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad())  
      
        
        