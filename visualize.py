import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cf import ItemBasedCF
from mf import MFRecommender

def load_data(data_path='ml-latest-small'):
    """
    Load MovieLens dataset
    
    Parameters:
        data_path: Dataset path
        
    Returns:
        ratings_df: Rating data
        movies_df: Movie data
    """
    ratings_file = os.path.join(data_path, 'ratings.csv')
    movies_file = os.path.join(data_path, 'movies.csv')
    
    ratings_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)
    
    return ratings_df, movies_df

def item_based_cf_results(ratings_df):
    """
    Run item-based collaborative filtering model and get results
    
    Parameters:
        ratings_df: Rating data
    """
    print("Training item-based collaborative filtering model...")
    cf_model = ItemBasedCF()
    cf_model.fit(ratings_df)
    
    # Find the 10 most similar movies for the given movie IDs
    target_movies = [260, 1407, 4993]
    
    print("\nItem-based collaborative filtering model results:")
    for movie_id in target_movies:
        similar_movies = cf_model.get_similar_movies(movie_id, top_n=10)
        print(f"10 most similar movies to movie {movie_id}: {similar_movies}")
    
    return cf_model

def matrix_factorization_results(ratings_df):
    """
    Run matrix factorization model and get results
    
    Parameters:
        ratings_df: Rating data
    """
    print("\nTraining matrix factorization model...")
    mf_model = MFRecommender(n_factors=20, n_epochs=5)  # Reduce training epochs to speed up demonstration
    mf_model.fit(ratings_df)
    
    # Recommend top 10 movies for given users
    target_users = [1, 2, 3]
    
    print("\nMatrix factorization model results:")
    for user_id in target_users:
        recommended_movies = mf_model.recommend_for_user(user_id, top_n=10)
        print(f"10 movies recommended for user {user_id}: {recommended_movies}")
    
    return mf_model

def visualize_data_and_results(ratings_df, movies_df, cf_model, mf_model):
    """
    Create data visualizations to analyze dataset and model results
    
    Parameters:
        ratings_df: Rating data
        movies_df: Movie data
        cf_model: Trained item-based collaborative filtering model
        mf_model: Trained matrix factorization model
    """
    plt.style.use('ggplot')
    
    # Create a directory for saving visualizations
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Rating Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_df['rating'], bins=10, kde=True)
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.savefig('visualizations/rating_distribution.png')
    plt.close()
    
    # 2. User Activity Analysis
    user_activity = ratings_df['userId'].value_counts().reset_index()
    user_activity.columns = ['userId', 'rating_count']
    
    plt.figure(figsize=(12, 6))
    sns.histplot(user_activity['rating_count'], bins=50, kde=True)
    plt.title('Distribution of User Activity')
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Number of Users')
    plt.savefig('visualizations/user_activity.png')
    plt.close()
    
    # 3. Movie Popularity Analysis
    movie_popularity = ratings_df['movieId'].value_counts().reset_index()
    movie_popularity.columns = ['movieId', 'rating_count']
    
    plt.figure(figsize=(12, 6))
    sns.histplot(movie_popularity['rating_count'], bins=50, kde=True)
    plt.title('Distribution of Movie Popularity')
    plt.xlabel('Number of Ratings per Movie')
    plt.ylabel('Number of Movies')
    plt.savefig('visualizations/movie_popularity.png')
    plt.close()
    
    # 4. Top Movies by Average Rating (with minimum 100 ratings)
    movie_stats = ratings_df.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'rating_count']
    popular_movies = movie_stats[movie_stats['rating_count'] >= 100].sort_values('avg_rating', ascending=False).head(20)
    popular_movies = popular_movies.reset_index().merge(movies_df[['movieId', 'title']], on='movieId')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='avg_rating', y='title', data=popular_movies)
    plt.title('Top 20 Movies by Average Rating (min 100 ratings)')
    plt.xlabel('Average Rating')
    plt.tight_layout()
    plt.savefig('visualizations/top_rated_movies.png')
    plt.close()
    
    # 5. Visualize Item-Based CF Results
    target_movie_id = 260  # Star Wars
    similar_movies = cf_model.get_similar_movies(target_movie_id, top_n=10)
    
    # Get titles for similar movies
    similar_movies_df = movies_df[movies_df['movieId'].isin(similar_movies)]
    similar_movies_df = similar_movies_df.set_index('movieId').loc[similar_movies].reset_index()
    
    target_movie_title = movies_df[movies_df['movieId'] == target_movie_id]['title'].values[0]
    
    plt.figure(figsize=(14, 8))
    plt.barh(similar_movies_df['title'], np.arange(1, len(similar_movies) + 1)[::-1])
    plt.title(f'Top 10 Movies Similar to "{target_movie_title}"')
    plt.xlabel('Similarity Rank')
    plt.tight_layout()
    plt.savefig('visualizations/similar_movies.png')
    plt.close()
    
    # 6. Visualize Matrix Factorization Latent Factors
    # Create a small heatmap of user factors for the first 10 users and factors
    if hasattr(mf_model.model, 'user_factors') and hasattr(mf_model.model, 'item_factors'):
        plt.figure(figsize=(12, 8))
        user_factors = mf_model.model.user_factors.detach().cpu().numpy()
        sns.heatmap(user_factors[:10, :10], cmap='viridis', annot=True, fmt='.2f')
        plt.title('User Latent Factors (First 10 Users x 10 Factors)')
        plt.xlabel('Factor Index')
        plt.ylabel('User Index')
        plt.savefig('visualizations/user_factors.png')
        plt.close()
        
        # Visualization of item factors for the first 10 items and factors
        plt.figure(figsize=(12, 8))
        item_factors = mf_model.model.item_factors.detach().cpu().numpy()
        sns.heatmap(item_factors[:10, :10], cmap='viridis', annot=True, fmt='.2f')
        plt.title('Item Latent Factors (First 10 Movies x 10 Factors)')
        plt.xlabel('Factor Index')
        plt.ylabel('Movie Index')
        plt.savefig('visualizations/item_factors.png')
        plt.close()
    
    print("\nVisualizations have been saved to the 'visualizations' directory")

def visualize_similar_movies(movies_df, cf_model, movie_id):
    similar_movies = cf_model.get_similar_movies(movie_id, top_n=10)
    similar_movies_df = movies_df[movies_df['movieId'].isin(similar_movies)]
    similar_movies_df = similar_movies_df.set_index('movieId').loc[similar_movies].reset_index()
    target_movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
    
    plt.figure(figsize=(14, 8))
    plt.barh(similar_movies_df['title'], np.arange(1, len(similar_movies) + 1)[::-1])
    plt.title(f'Top 10 Movies Similar to "{target_movie_title}"')
    plt.xlabel('Similarity Rank')
    plt.tight_layout()
    plt.savefig(f'visualizations/similar_movies_{movie_id}.png')
    plt.close()

def visualize_user_recommendations(movies_df, mf_model, user_id):
    """
    Visualize movie recommendations for a specific user
    
    Parameters:
        movies_df: Movie data
        mf_model: Trained matrix factorization model
        user_id: User ID
    """
    recommended_movies = mf_model.recommend_for_user(user_id, top_n=10)
    recommended_movies_df = movies_df[movies_df['movieId'].isin(recommended_movies)]
    # Ensure ordering matches recommendation order
    recommended_movies_df = recommended_movies_df.set_index('movieId').loc[recommended_movies].reset_index()
    
    plt.figure(figsize=(14, 8))
    plt.barh(recommended_movies_df['title'], np.arange(1, len(recommended_movies) + 1)[::-1])
    plt.title(f'Top 10 Movie Recommendations for User {user_id}')
    plt.xlabel('Recommendation Rank')
    plt.tight_layout()
    plt.savefig(f'visualizations/user_{user_id}_recommendations.png')
    plt.close()

def display_user_rating_history(ratings_df, movies_df, user_id, top_n=10):
    """
    Display the rating history for a specific user
    
    Parameters:
        ratings_df: Rating data
        movies_df: Movie data
        user_id: User ID
        top_n: Number of ratings to display
    """
    # Get user ratings
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    # Sort by rating (descending)
    user_ratings = user_ratings.sort_values('rating', ascending=False)
    
    # Get top N ratings
    top_ratings = user_ratings.head(top_n)
    
    # Merge with movie data to get titles
    top_ratings_with_titles = top_ratings.merge(movies_df[['movieId', 'title']], on='movieId')
    
    print(f"\nTop {top_n} ratings for User {user_id}:")
    for _, row in top_ratings_with_titles.iterrows():
        print(f"Movie: {row['title']}, Rating: {row['rating']}")

# Load data
ratings_df, movies_df = load_data()
print(f"Loaded {len(ratings_df)} rating records and {len(movies_df)} movies")
    
# Run item-based collaborative filtering model
cf_model = item_based_cf_results(ratings_df)
    
# Run matrix factorization model
mf_model = matrix_factorization_results(ratings_df)

# Create visualizations
visualize_data_and_results(ratings_df, movies_df, cf_model, mf_model)

# Visualize similar movies for movie IDs 260, 1407, and 4993
visualize_similar_movies(movies_df, cf_model, 260)
visualize_similar_movies(movies_df, cf_model, 1407)
visualize_similar_movies(movies_df, cf_model, 4993)

# Display rating history for users 1, 2, and 3
display_user_rating_history(ratings_df, movies_df, 1)
display_user_rating_history(ratings_df, movies_df, 2)
display_user_rating_history(ratings_df, movies_df, 3)

# Visualize recommendations for users 1, 2, and 3
visualize_user_recommendations(movies_df, mf_model, 1)
visualize_user_recommendations(movies_df, mf_model, 2)
visualize_user_recommendations(movies_df, mf_model, 3)
    
print("\nTask completed!")