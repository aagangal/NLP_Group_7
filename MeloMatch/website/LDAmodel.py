import numpy as np
import pandas as pd 
import nltk
import re
import spacy
from gensim import corpora, models
import gensim
from gensim.matutils import hellinger
from scipy.spatial.distance import cosine
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('vader_lexicon')

df = pd.read_csv(r'C:\Users\anaya\OneDrive\Desktop\Anay Masters\Intro to NLP\NLP_Group_7\FullyProcessedDataset.csv')

tokenized_corpus = [text.split() for text in df['Lemmatized_Lyrics']]

dictionary = corpora.Dictionary(tokenized_corpus)
corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]

lda_model = gensim.models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=35, random_state = 42)

song_topic_distribution = [lda_model[doc] for doc in corpus]