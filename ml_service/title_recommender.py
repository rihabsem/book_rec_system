from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

books = pd.read_csv("../data/books_clean.csv")
tfidf_title = TfidfVectorizer(stop_words="english")
tfidf_title_matrix = tfidf_title.fit_transform(books["title"])
similarity_title = cosine_similarity(tfidf_title_matrix)

def recommend_by_title(book_title):
    idx = books[books["title"] == book_title].index[0]
    sim_scores = list(enumerate(similarity_title[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5

    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices][["title","authors","categories"]]


print(recommend_by_title("harry potter"))

