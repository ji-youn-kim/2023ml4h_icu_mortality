import os
import torch
import torch.nn as nn
import pandas as pd


class EHRDataset(nn.Module):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        if self.split == 'train':
            data = pd.read_pickle(os.path.join(self.args.input_path, 'X_train_rnn.pkl'))
        else:
            data = pd.read_pickle(os.path.join(self.args.input_path, 'X_test_rnn.pkl'))
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)

    def collator(self, samples):
        x_cat = [s["x_cat"] for s in samples]
        x_cont = [s["x_cont"] for s in samples]
        s_len = [len(s["x_cat"]) for s in samples]
        labels = [s["label"] for s in samples]

        x_cat = nn.utils.rnn.pad_sequence(x_cat, batch_first=True) #(B, S, F_c)
        x_cont = nn.utils.rnn.pad_sequence(x_cont, batch_first=True) #(B, S, F_n)

        batch_input = {"x_cat": x_cat, #(B, S, #cat_features)
            "x_cont": x_cont, #(B, S, #num_features)
            "s_len": s_len}
        
        return {
            "input": batch_input,
            "labels": labels
        }


    def __getitem__(self, index):
        pack = {
            "x_cat": torch.tensor([[self.data[index]["ADMISSION_TYPE"], self.data[index]["ADMISSION_LOCATION"], self.data[index]["DIAGNOSIS"], i[1]] for i in self.data[index]["CHARTEVENTS"]]), #(S, #features)
            "x_cont": torch.tensor([[i[0], i[2]] for i in self.data[index]["CHARTEVENTS"]]),
            "label": self.data[index]["LABEL"]
        }
        return pack