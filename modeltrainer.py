import torch 
import torch.nn as nn
import torch.optim as optim
import copy
import sys
from dataloaders import DataLoaders

class ModelTrainer:
    def __init__(self,model,epochs,data):
        self.model = model
        self.epochs = epochs
        self.data = data
    
    def train(self):
        best_model = copy.deepcopy(self.model)
        lowest_test_loss = sys.float_info.max
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        train_loader, test_loader = DataLoaders(self.data).load()

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, targets, lengths in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs, lengths)
                loss = criterion(outputs.squeeze(-1), targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets, lengths in test_loader:
                    outputs = self.model(inputs, lengths)
                    loss = criterion(outputs.squeeze(-1), targets)
                    test_loss += loss.item()
            
            if test_loss < lowest_test_loss:
                best_model = copy.deepcopy(self.model)
                lowest_test_loss = test_loss
            
            print(epoch, test_loss)

        return best_model