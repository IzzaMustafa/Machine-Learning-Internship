import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

# Loading Files

movies = pd.read_csv("movies_.csv")
ratings = pd.read_csv("ratings_.csv")

# Setting a Target User (For Evaluation)

target = 30  

user_ratings = ratings[ratings["userId"] == target]

# Splitting Data

train, test = train_test_split(
    user_ratings, test_size= 0.2, random_state= 42
)

# Creating User-Item matrix using training data (only)

User_Item_Matrix = ratings[~ratings.index.isin(test.index)].pivot_table(
    index="userId", 
    columns="movieId", 
    values="rating"
)
User_Item_Matrix = User_Item_Matrix.fillna(0)
User_Similarity = cosine_similarity(User_Item_Matrix)

# Creating Item-Item matrix using training data (only)
Item_Similarity = cosine_similarity(User_Item_Matrix.T)

# Converting to Data Frames

us_df = pd.DataFrame(User_Similarity, index= User_Item_Matrix.index, columns= User_Item_Matrix.index)
is_df = pd.DataFrame(Item_Similarity, index= User_Item_Matrix.columns, columns= User_Item_Matrix.columns)

# Finding Users similar to Target User
similar_users = us_df[target].sort_values(ascending= False).drop(target)

# Rated Movies by the Target User
rated_movies = User_Item_Matrix.loc[target][User_Item_Matrix.loc[target] > 0].index
item_rated_movies = train.set_index("movieId")["rating"]

# Unseen Movies by the Target User
unseen_movies = User_Item_Matrix.drop(columns= rated_movies)


# Reindexing Similar Users
similar_users = similar_users.reindex(unseen_movies.index, fill_value= 0)

# Similarity Scores
scores = (unseen_movies.T.dot(similar_users)) /similar_users.sum()

# Item Similarity Scores
item_scores = {}
for movie_id, rating in item_rated_movies.items():
    similar_movies = is_df[movie_id].drop(movie_id)
    for sim_movie, sim_score in similar_movies.items():
        if sim_movie not in item_rated_movies.index:
            item_scores[sim_movie] = item_scores.get(sim_movie, 0) + sim_score * rating

# Recommended Movies with IDs & Titles (For Ease)
recommended = scores.sort_values(ascending= False)

recommended_movies = movies[movies["movieId"].isin(recommended.index)]
recommended_movies = recommended_movies[["movieId", "title"]]
print("User-Based Recommended Movies: \n", recommended_movies)

# Creating a Function (For User-Item Matrix) to Evaluate Precision
def precision_at_k(recommended_movies, test_ratings, k = 5, threshold = 4):
    relavent = test_ratings[test_ratings >= threshold].index
    top_k = recommended_movies[:k]

    hit = len(set(relavent) & set(top_k))
    return hit/k

# Selecting the number of movies from recommended list
recommended_movies = recommended.index[:10]

# Using test set: getting actual ratings indexed by movieId
test_ratings = test.set_index("movieId")["rating"]

# Evaluating Precision @ K
print("\nPrecision @ 5 (User-Based): ", precision_at_k(recommended_movies, test_ratings, k=5)) # This metric can also be averaged across users for a broader system evaluation

# Recommend Movies for Item-Item Matrix with IDs & Titles
recommended_item = pd.Series(item_scores).sort_values(ascending=False)

recommended_movies_item = movies[movies["movieId"].isin(recommended_item.index)]
recommended_movies_item = recommended_movies_item[["movieId", "title"]]
print("\nItem-Based Recommended Movies: \n", recommended_movies_item)

# Selecting the number of movies from recommended_item list
recommended_movies_item = recommended_item.index[:10]

# Evaluating Precision @ K
print("\nPrecision @ 5 (Item-Based): ", precision_at_k(recommended_movies_item, test_ratings, k=5))

# Apply SVD (choose number of latent factors, e.g., 20)
svd = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd.fit_transform(User_Item_Matrix)

# Get latent factors for users & movies
user_factors = latent_matrix
movie_factors = svd.components_.T 

# Predict ratings for the target user
target_index = User_Item_Matrix.index.get_loc(target)
predicted_ratings = np.dot(user_factors[target_index], movie_factors.T)

# Converting predictions to Series with movieIds
pred_ratings = pd.Series(predicted_ratings, index=User_Item_Matrix.columns)

# Remove already seen movies
pred_ratings = pred_ratings.drop(rated_movies)

# Get top-N recommendations
recommended_svd = pred_ratings.sort_values(ascending=False)

# Show with titles
recommended_movies_svd = movies[movies["movieId"].isin(recommended_svd.index)]
print("\nSVD-Based Recommended Movies: \n", recommended_movies_svd)

# Evaluating Precision @ K
print("\nPrecision @ 5 (SVD):", precision_at_k(recommended_svd.index[:5], test_ratings, k=5))