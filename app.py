import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Load the dataset
data = pd.read_csv('amazon_product.csv')

# Ensure 'ImageURL' column exists
if 'ImageURL' not in data.columns:
    st.error("The 'ImageURL' column is missing in the dataset.")
else:
    # Remove unnecessary columns
    data = data.drop('id', axis=1)

    # Handle missing values in 'ImageURL'
    data['ImageURL'] = data['ImageURL'].fillna('img2.png') # Update to the correct path

    # Define tokenizer and stemmer
    stemmer = SnowballStemmer('english')
    def tokenize_and_stem(text):
        tokens = nltk.word_tokenize(text.lower())
        stems = [stemmer.stem(t) for t in tokens]
        return stems

    # Create stemmed tokens column
    data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

    # Define TF-IDF vectorizer and cosine similarity function
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
    def cosine_sim(text1, text2):
        text1_concatenated = ' '.join(text1)
        text2_concatenated = ' '.join(text2)
        tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
        return cosine_similarity(tfidf_matrix)[0][1]

    # Define search function
    def search_products(query):
        query_stemmed = tokenize_and_stem(query)
        data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
        results = data.sort_values(by=['similarity'], ascending=False).head(4)[['Title', 'Description', 'Category', 'ImageURL']]
        return results

    # Web app
    img = Image.open('img.png')
    st.image(img, width=600)

    st.title("Search Engine and Product Recommendation")
    query = st.text_input("Enter Product or Category Name")
    submit = st.button('Search')
    if submit:
        res = search_products(query)
        for idx, row in res.iterrows():
            # Load the image from the URL or use the default image
            image_url = row['ImageURL']
            try:
                if image_url.startswith('http'):
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))
                else:
                    image = Image.open(image_url)
                st.image(image, width=200)
            except Exception as e:
                st.error(f"Could not load image from {image_url}: {e}")
                # Provide a default image if the URL fails
                st.image('default_image.png', width=200)  # Ensure this path is correct

            st.write(f"**Title**: {row['Title']}")
            st.write(f"**Description**: {row['Description']}")
            st.write(f"**Category**: {row['Category']}")
        st.write(res)
