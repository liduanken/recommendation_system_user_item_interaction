import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedCF:
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity = None
        self.movie_ids = None
    
    def fit(self, ratings_df):
        """
        Build item-based collaborative filtering model
        
        Parameters:
            ratings_df: DataFrame containing user ratings in format (userId, movieId, rating)
        """
        # Create user-item rating matrix
        pivot_table = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
        
        # Fill missing values with 0
        self.user_item_matrix = pivot_table.fillna(0)
        
        # Calculate item similarity matrix (using cosine similarity)
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        # Save movie ID order for later lookup
        self.movie_ids = self.user_item_matrix.columns.tolist()
    
    def get_similar_movies(self, movie_id, top_n=10):
        """
        Get N most similar movies to the given movie
        
        Parameters:
            movie_id: Movie ID
            top_n: Number of similar movies to return
            
        Returns:
            List containing similar movie IDs
        """
        if movie_id not in self.movie_ids:
            print(f"Movie ID {movie_id} not in training data")
            return []
        
        # Get movie index in matrix
        idx = self.movie_ids.index(movie_id)
        
        # Get similarities to this movie
        movie_similarities = self.item_similarity[idx]
        
        # Get top_n most similar movies (excluding itself)
        similar_indices = np.argsort(movie_similarities)[::-1][1:top_n+1]
        
        # Convert back to movie IDs
        similar_movies = [self.movie_ids[i] for i in similar_indices]
        
        return similar_movies 