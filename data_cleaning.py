import pandas as pd
import re

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text
books = pd.read_csv("../data/Books.csv")
print("books read")
ratings = pd.read_csv("../data/Ratings.csv")
print("ratings read")
print(books.head())
print(ratings.head())
print(books.info())
print(ratings.info())

books["Author"] = books["Author"].fillna("")
#drop duplicates
books = books.drop_duplicates(subset=["ISBN","Title","Author"])
ratings = ratings.drop_duplicates(subset=["User-ID","ISBN"])
list_books = [col for col in books.columns if col != 'Year' and col != 'ISBN']
for colonne in list_books:
    books[colonne] = books[colonne].apply(clean_text)

#saving the documents
books.to_csv("../data/books_clean.csv",index=False)
ratings.to_csv("../data/ratings_clean.csv",index=False)