from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

books = pd.read_csv("../data/books_clean.csv")
tfidf_category = TfidfVectorizer(stop_words="english")
tfidf_category_matrix = tfidf_category.fit_transform(books["categories"])
similarity_category = cosine_similarity(tfidf_category_matrix)

def recommend_by_category(book_categories):
    idx = books[books["categories"] == book_categories].index[0]
    sim_scores = list(enumerate(similarity_category[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5

    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices][["title","authors","categories"]]


print(recommend_by_category("juvenile fiction"))

