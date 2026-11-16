import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

df_book = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv')
df_book_tags = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv')
df_ratings = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv')
df_tags = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv')

df_book.drop(columns=['isbn', 'isbn13'],inplace=True)
df_book['original_title'].fillna(df_book['title'])
df_book['language_code'].fillna('unknown')
df_book['original_publication_year'].fillna(0.0)
# cleaning the df_book_tags dataframe
df_book_tags = df_book_tags.drop_duplicates(subset=['goodreads_book_id', 'tag_id'])

# removing tags that aren't relevant for recommendation in df_tags
exact_noise = {
    "to-read", "currently-reading", "favorites", "favourites", "owned", "books-i-own", "library", "kindle", "ebook", "ebooks", "default",
    "audiobook", "audiobooks", "re-read", "reread", "dnf", "did-not-finish","borrowed", "calibre", "general", "shelfari-favorites", "reviewed"
}
noise_patterns = (
    r'to-read|to-be-read|tbr|owned|wish-list|wishlist|library|'
    r'read-in-|reads-|reading|favorite|favourite|book-club|series|'
    r'^[^a-z]*$'  # removes tags that are just numbers/years
)

df_tags['tag_name'] = df_tags['tag_name'].str.lower().str.strip().str.strip('-')
# due to this cleaning of the tag_name, there'll be duplicated values in df_tags but that'll be dealt with after

df_tags = df_tags[~df_tags['tag_name'].isin(exact_noise) & ~df_tags['tag_name'].str.contains(noise_patterns,regex=True)]
df_books_with_tags = pd.merge(df_book_tags, df_tags, on='tag_id', how='left') 
df_books_with_tags.drop(columns=['tag_id'], inplace=True)

# dealing with the duplicates that are inherited from df_tags due to the cleaning of tag_name
df_books_with_tags = df_books_with_tags.groupby(['goodreads_book_id', 'tag_name'],as_index=False)['count'].sum()

#collecting the top 10 most popular tags for each book
df_top10_tags = df_books_with_tags[df_books_with_tags.groupby('goodreads_book_id')['count'].rank(method='first', ascending=False) <= 10]

# merging the top 10 tags attributed to a book in a single row
df_row_book_tag = df_top10_tags.groupby('goodreads_book_id', as_index=False).agg({'tag_name': ', '.join})

df_books_info = pd.merge(df_book, df_row_book_tag, on='goodreads_book_id', how='left')
df_books_info.drop(columns=['goodreads_book_id', 'best_book_id', 'work_id', 'books_count', 'original_title', 'work_text_reviews_count', 'work_ratings_count'], inplace=True)

v = df_books_info['ratings_count']
m = 40000 # based on the 75th percentile of the quantity of ratings for one book 
R = df_books_info['average_rating']
C = df_ratings['rating'].mean()
df_books_info['weighted_average_rating'] = (v/(v+m))*R+(m/(v+m))*C # IMDb weighted rating formula

df_books_linear = df_books_info[['book_id', 'weighted_average_rating', 'tag_name', 'authors']].copy()
df_ratings_sampled = df_ratings.sample(100000, random_state=1) # df_ratings has more than 6 million rows, to avoid crashing, sampling
del df_ratings_sampled['user_id']
df_linear_regression = pd.merge(df_ratings_sampled, df_books_linear, on='book_id', how='left')
del df_linear_regression['book_id'] # book_id was only needed to do the merge
df_linear_regression['tag_name'] = df_linear_regression['tag_name'].str.split(',').str[0].str.strip() # only uses the first tag of the book

df_full_train, df_test = train_test_split(df_linear_regression, test_size=0.2, random_state=1)
y_full_train = df_full_train.rating.values
y_test = df_test.rating.values

df_full_train_clean = df_full_train.drop(columns='rating')
df_test_clean = df_test.drop(columns='rating')

dv = DictVectorizer(sparse=True)
full_train_dict = df_full_train_clean.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)
test_dict = df_test_clean.to_dict(orient='records')
X_test = dv.transform(test_dict)
model = Ridge(alpha=100.0, random_state=1)
model.fit(X_full_train, y_full_train)
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Ridge Model (alpha = 100.0)--->RMSE = {rmse:.4f}")

from pathlib import Path

# Automatically finds the exact folder where train.py is
project_dir = Path(__file__).resolve().parent
output_file = project_dir / "model.bin"

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)
print(f"Saved model.bin successfully to: {output_file}")
