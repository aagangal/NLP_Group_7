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


    
user_song = int((input("Enter song index: ")))
print(user_song)
df.head(11)
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(lyrics):
    sentiment = analyzer.polarity_scores(lyrics)
    sentiment_score = sentiment['compound']
    return sentiment_score

sentiment = get_sentiment_score(df['Lemmatized_Lyrics'].iloc[user_song])
print(sentiment)

SSF_list = []
for i in range(df.shape[0]):
    SSF_list.append(abs(df['Sentiment'].iloc[i] - sentiment))

df['SSF'] = SSF_list
df.head(21)


tfidf_vectorizer = TfidfVectorizer()
CSF_list = []
for i in range(df.shape[0]):
    lyrics_matrix = tfidf_vectorizer.fit_transform([df['Lemmatized_Lyrics'].iloc[user_song], df['Lemmatized_Lyrics'].iloc[i]])
    lyrics_similarity = cosine_similarity(lyrics_matrix)
    CSF_list.append(1 - (lyrics_similarity[0][1]))
df['CSF'] = CSF_list
df.head(11)

def calculate_hellinger_distance(song_dist_1, song_dist_2):
    return hellinger(song_dist_1, song_dist_2)

input_song_index = user_song

input_song_dist = song_topic_distribution[input_song_index]

hellinger_distances = []

for i, song_dist in enumerate(song_topic_distribution):

    distance = calculate_hellinger_distance(input_song_dist, song_dist)
    hellinger_distances.append((distance))

df['HDF'] = hellinger_distances
df.head(21)

df['SF'] = df['SSF'] + df['CSF'] + df['HDF']
Recommendationdf = df.nsmallest(6,["SF"])
Recommendationdf.drop(user_song, axis = 0, inplace = True)

RecommendationDF = Recommendationdf[['ALink', 'SName']]
RecommendationDF.reset_index(inplace = True)

return RecommendationDF



