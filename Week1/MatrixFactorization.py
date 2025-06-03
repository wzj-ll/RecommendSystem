import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def load_data(file_path: str):
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    interaction = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    

class MatrixFactorization(nn.Module):
    