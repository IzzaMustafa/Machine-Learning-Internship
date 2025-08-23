# Movie Recommendation System (Collaborative Filtering)

This project builds a movie recommendation system using the **MovieLens dataset**.  
I implemented three approaches to recommend movies for a target user:

### 🔹 Approaches
1. **User-Based Collaborative Filtering (UBCF)** – Finds similar users using cosine similarity.  
2. **Item-Based Collaborative Filtering (IBCF)** – Finds similar movies based on user ratings.  
3. **Matrix Factorization (SVD)** – Uses latent factors to predict unseen ratings.  

### 📊 Evaluation
I used **Precision@K (K=5)** as the evaluation metric.  

For the chosen **Target User = 30**, the results were:  
- **User-Based CF:** Precision@5 = **1.0**  
- **Item-Based CF:** Precision@5 = **0.8**  
- **SVD:** Precision@5 = **0.4**  

⚠️ **Note:** These values vary depending on the target user.  
- For **User = 50**, Precision@5 was **0.4** for all methods.  
- For **User = 40**, Precision@5 was **0.2** for User-Based & SVD, but **0.4** for Item-Based.  

### 💡 Key Insights
- User-based filtering performed the best for User 30.  
- Item-based was also strong, while SVD gave lower precision.  
- Recommendation quality changes for different users, showing the importance of evaluating across multiple users.  
