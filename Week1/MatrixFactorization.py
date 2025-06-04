import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def load_data(file_path: str):
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    df.drop('timestamp', axis=1, inplace=True)
    return df

class RatingDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame):
        super().__init__()
        self.users = torch.tensor(dataframe['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(dataframe['item_id'].values, dtype=torch.long)
        # 在神经网络中，Python 的 float 默认使用float32的数据类型
        self.ratings = torch.tensor(dataframe["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        return self.users[index], self.items[index], self.ratings[index]

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
    
    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        dot_product = (user_vecs * item_vecs).sum(dim=1)
        return dot_product
    
def train(model, dataloader, epochs=5, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for user, item, rating in dataloader:
            pred = model(user, item)
            loss = criterion(pred, rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    df = load_data("./ml-100k/u.data")

    num_users = df["user_id"].count()
    num_items = df["item_id"].count()
    
    dataset = RatingDataset(df)

    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = MatrixFactorization(num_users, num_items)
    train(model, dataloader, epochs=20, lr=0.05)