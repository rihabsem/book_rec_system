from sklearn.feature_extraction.text import TfidfVectorizer #pour transformer du texte en vecteur comprehensible par le model
from sklearn.metrics.pairwise import cosine_similarity #pour mesurer a quel point deux livres sont proches l'un de l'autre
import pandas as pd

books = pd.read_csv("../data/books_clean.csv")
tfidf_author = TfidfVectorizer(stop_words="english")
tfidf_author_matrix = tfidf_author.fit_transform(books["authors"]) #transforme la colonne auteur en vecteur
similarity_author = cosine_similarity(tfidf_author_matrix)

def recommend_by_author(book_author):
    idx = books[books["authors"] == book_author].index[0] #recupere l'indice de l'auteur sur qui on fait l'étude
    sim_scores = list(enumerate(similarity_author[idx])) #applique la matrice de similarité sur chaque auteur du dataset et retourne le résultat en liste de (indice_livre, score similarité)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #trie la liste d'ordre decroissant
    sim_scores = sim_scores[1:6]  # top 5 sans le livre lui meme

    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices][["title","authors","categories"]]


print(recommend_by_author("stephen r donaldson"))

