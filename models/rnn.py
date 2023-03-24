import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


class EHRRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Embedding Order: ADMISSION_TYPE, ADMISSION_LOCATION, DIAGNOSIS, ITEMID
        self.embedding_sizes = [6, 11, 4816, 1116]
        self.embeddings = nn.ModuleList([nn.Embedding(categories, self.args.embed_dim, padding_idx=0) for categories in self.embedding_sizes])

        input_size = self.args.embed_dim * len(self.embedding_sizes) + 2
        self.gru = nn.GRU(input_size, self.args.hidden_dim, self.args.num_layers, batch_first=True) 
        
        self.proj1 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim // 2) 
        self.proj2 = nn.Linear(self.args.hidden_dim // 2, self.args.hidden_dim // 4) 
        self.final = nn.Linear(self.args.hidden_dim // 4, self.args.n_class) 


    def forward(self, x_cat, x_cont, s_len):

        # x_cat shape: (B, S, #features)
        B,S,F = x_cat.shape
        x1 = [e(x_cat[:, :, i]) for i, e in enumerate(self.embeddings)] # (B, S, emb) * #features
        x1 = torch.cat(x1, 2) # (B, S, #features*emb)
        # Concat categorical / numerical columns
        x = torch.cat([x1, x_cont], 2) # (B, S, #features_cat*emb + #features_cont)
        x = nn.utils.rnn.pack_padded_sequence(x, batch_first=True, lengths=s_len, enforce_sorted=False)
        
        output, _ = self.gru(x.float()) 
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # output: (B, S, h_out)

        # Mean pooling & Project Output
        output = torch.mean(output, 1) # (B, h_out)
        output = self.proj1(output)
        output = self.proj2(output)
        output = self.final(output) # (B, n_class)

        return output
