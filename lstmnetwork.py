import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pathlib import Path

class LSTMNetwork(nn.Module):
    def __init__(self):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=32, batch_first=True, num_layers=4)
        self.linear = nn.Linear(32, 1)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        linear_output = self.linear(unpacked_output)
        return linear_output
    
    def save(self,path):
        torch.save(self, path)
    
    def load(self,path):
        if Path(path).exists():
            self = torch.load(path, weights_only=False)

        return self