from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pprint import pprint
import numpy as np

word_data = pickle.load(file=open('your_word_data.pkl', 'rb'))

tfidf = TfidfVectorizer(stop_words='english')

tfidf = tfidf.fit(word_data)
print tfidf.get_feature_names()[34597]

tfidf_matrix = tfidf.transform(word_data)
print tfidf_matrix.shape
