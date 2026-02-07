#1 decode
#2 generate embeddigs if necessary
#3 train the model
import pandas as pd

books = pd.read_csv("../data/books_clean.csv")
books.info()

