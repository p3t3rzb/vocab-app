import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_tensor = torch.tensor(sample['input'], dtype=torch.float32)
        output_tensor = torch.tensor(sample['output'], dtype=torch.float32)
        return input_tensor, output_tensor

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    outputs = [item[1] for item in batch]
    
    padded_inputs = pad_sequence([seq.clone().detach() for seq in inputs], batch_first=True, padding_value=0)
    lengths = torch.tensor([len(seq) for seq in inputs])
    padded_outputs = pad_sequence([seq.clone().detach() for seq in outputs], batch_first=True, padding_value=0)
    
    return padded_inputs, padded_outputs, lengths

class DataLoaders:
    def __init__(self,data):
        self.data = data

    def load(self):
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)

        train_dataset = SequenceDataset(train_data)
        test_dataset = SequenceDataset(test_data)

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)

        return train_loader, test_loader