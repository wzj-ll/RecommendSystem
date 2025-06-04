import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

def load_data(file_path : str):
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    df.drop('timestamp', axis=1, inplace=True)
    df["user_id"] -= 1
    df["item_id"] -= 1
    
    return df

class PairwiseDataset(Dataset):
    def __init__(self, df:pd.DataFrame, num_items, num_negatives=1):
        super().__init__()
        self.data = []
        user_item_set = set(zip(df.user_id, df.item_id))
        for u, i ,r in zip(df.user_id, df.item_id, df.rating):
            if r >= 4.0:
                self.data.append((u, i, 1))
                for _ in range(num_negatives):
                    j = random.randint(0, num_items - 1)
                    while (u, j) in user_item_set:
                        j = random.randint(0, num_items - 1)
                    self.data.append((u, j, 0))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        u, i, label = self.data[index]
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(label, dtype=torch.float32)
    
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
    
    def forward(self, user_ids, item_ids):
        self.user_vecs = self.user_embedding(user_ids)
        self.item_vecs = self.item_embedding(item_ids)
        scores = (self.user_vecs * self.item_vecs).sum(dim=1)
        return torch.sigmoid(scores)
    
