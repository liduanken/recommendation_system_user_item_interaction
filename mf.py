import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MovieLensDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = ratings_df['userId'].values
        self.items = ratings_df['movieId'].values
        self.ratings = ratings_df['rating'].values
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float)
        )

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super(MatrixFactorization, self).__init__()
        
        # Global average rating
        self.mu = nn.Parameter(torch.zeros(1))
        
        # User and item biases
        self.user_bias = nn.Parameter(torch.zeros(n_users))
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        
        # User and item latent factors
        self.user_factors = nn.Parameter(torch.randn(n_users, n_factors) * 0.01)
        self.item_factors = nn.Parameter(torch.randn(n_items, n_factors) * 0.01)
        
    def forward(self, user, item):
        # Calculate predicted rating
        pred = self.mu + self.user_bias[user] + self.item_bias[item]
        pred += (self.user_factors[user] * self.item_factors[item]).sum(dim=1)
        return pred

class MFRecommender:
    def __init__(self, n_factors=20, lr=0.005, reg=0.02, n_epochs=20, batch_size=64):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_id_map = {}
        self.item_id_map = {}
        self.inv_user_id_map = {}
        self.inv_item_id_map = {}
        
    def _create_id_mappings(self, ratings_df):
        # Create user ID mapping
        unique_users = ratings_df['userId'].unique()
        self.user_id_map = {user_id: i for i, user_id in enumerate(unique_users)}
        self.inv_user_id_map = {i: user_id for user_id, i in self.user_id_map.items()}
        
        # Create item ID mapping
        unique_items = ratings_df['movieId'].unique()
        self.item_id_map = {item_id: i for i, item_id in enumerate(unique_items)}
        self.inv_item_id_map = {i: item_id for item_id, i in self.item_id_map.items()}
        
        # Map IDs in the dataset
        mapped_df = ratings_df.copy()
        mapped_df['userId'] = mapped_df['userId'].map(self.user_id_map)
        mapped_df['movieId'] = mapped_df['movieId'].map(self.item_id_map)
        
        return mapped_df
    
    def fit(self, ratings_df):
        # Create ID mappings
        mapped_ratings = self._create_id_mappings(ratings_df)
        
        # Calculate global mean rating
        global_mean = mapped_ratings['rating'].mean()
        
        # Create dataset and dataloader
        dataset = MovieLensDataset(mapped_ratings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        self.model = MatrixFactorization(n_users, n_items, self.n_factors)
        self.model.to(self.device)
        
        # Set model to training mode
        self.model.train()
        
        # Initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Initialize global mean rating
        self.model.mu.data.fill_(global_mean)
        
        # Train model
        for epoch in range(self.n_epochs):
            running_loss = 0.0
            for user_ids, item_ids, ratings in dataloader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                # Forward pass
                predictions = self.model(user_ids, item_ids)
                
                # Calculate loss
                loss = nn.MSELoss()(predictions, ratings)
                
                # Add regularization term
                reg_loss = self.reg * (
                    self.model.user_bias[user_ids].pow(2).sum() + 
                    self.model.item_bias[item_ids].pow(2).sum() + 
                    self.model.user_factors[user_ids].pow(2).sum() + 
                    self.model.item_factors[item_ids].pow(2).sum()
                )
                loss += reg_loss
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {running_loss/len(dataloader):.4f}')
        
        # Set model to evaluation mode
        self.model.eval()
    
    def recommend_for_user(self, user_id, top_n=10):
        """
        Recommend movies for a given user
        
        Parameters:
            user_id: User ID
            top_n: Number of movies to recommend
            
        Returns:
            List of recommended movie IDs
        """
        if user_id not in self.user_id_map:
            print(f"User ID {user_id} not in training data")
            return []
        
        # Map user ID
        mapped_user_id = self.user_id_map[user_id]
        
        with torch.no_grad():
            # Calculate predicted ratings for all items
            user_tensor = torch.LongTensor([mapped_user_id] * len(self.item_id_map)).to(self.device)
            item_tensor = torch.LongTensor(list(range(len(self.item_id_map)))).to(self.device)
            
            predictions = self.model(user_tensor, item_tensor).cpu().numpy()
        
        # Get items with highest ratings
        top_item_indices = np.argsort(predictions)[::-1][:top_n]
        
        # Convert back to original item IDs
        recommended_items = [self.inv_item_id_map[idx] for idx in top_item_indices]
        
        return recommended_items 