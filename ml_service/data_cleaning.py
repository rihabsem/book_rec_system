import pandas as pd
import re
#nombre de lignes 6811
def text_cleaning(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()
    return text

books = pd.read_csv("../data/books.csv")
books.drop(columns=["subtitle", "thumbnail", ])
print("books read")
print(books.head())
books = books.drop(columns= "subtitle")
books = books.drop(columns= "thumbnail")
print(books.head())
print(books.info())

#traitement des données manquantes
    #donnée categoriel
books["authors"] = books["authors"].fillna("Unknown Author")
books["categories"] = books["categories"].fillna("Uncategorized")
books["description"] = books["description"].fillna("")
    #donnée numerique
books["published_year"] = books["published_year"].fillna(books["published_year"].median())
books["average_rating"] = books["average_rating"].fillna(books["average_rating"].median())
books["num_pages"] = books["num_pages"].fillna(books["num_pages"].median())
books["ratings_count"] = books["ratings_count"].fillna(books["ratings_count"].median())

books.info()

# text cleaning
books['title'] = books['title'].apply(text_cleaning)
books['authors'] = books['authors'].apply(text_cleaning)
books['categories'] = books['categories'].apply(text_cleaning)
books['description'] = books['description'].apply(text_cleaning)
books.info()
books.to_csv("../data/books_clean.csv",index=False)
