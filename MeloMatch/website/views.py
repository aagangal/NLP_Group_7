from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd 
import nltk
import re
import spacy
from bs4 import BeautifulSoup
from gensim import corpora, models
import gensim
from gensim.matutils import hellinger
from scipy.spatial.distance import cosine
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('vader_lexicon')

#from ipynb.fs.full.MM import RecommendSongs
from ipynb.fs.full.newtest import RecommendSongs, song_topic_distribution

views = Flask(name)

@views.route('/')
def index():

    return render_template('index.html')


@views.route('/search', methods=['GET', 'POST'])
def search():

    data = request.get_json()
    searchTerm = data['searchTerm']

    finalOutput = RecommendSongs(song_topic_distribution, searchTerm)

    return jsonify(finalOutput.to_dict(orient='records'))


if name == 'main':
    views.run(debug=True)