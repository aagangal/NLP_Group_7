# from flask import Blueprint

# views = Blueprint('views', __name__)

# @views.route('/')
# def home():
#     return "<h1>MeloMatch</h1>"
# from flask import Flask, render_template, request
# import pandas as pd

# views = Flask(__name__)

# @views.route('/', methods=['GET', 'POST'])
# def index():

#     data = {'Name': ['Alice', 'Bob', 'Charlie'],
#         'Age': [25, 30, 35],
#         'City': ['New York', 'Paris', 'London']}
#     df = pd.DataFrame(data)

#     #songs = {}
#     if request.method == 'POST':
#         songs = df
#     return render_template('index.html', songs=songs)

# if __name__ == '__main__':
#     views.run(debug=True)
#import MM
import MeloMatch
from flask import Flask, render_template, request, jsonify

views = Flask(__name__)

@views.route('/')
def index():
    return render_template('index.html')

@views.route('/search', methods=['POST'])
def search():
    #Dummy data to simulate a response from a Python script
    # data = [
    #     {"Title": "Song One", "Artist": "Artist One"},
    #     {"Title": "Song Two", "Artist": "Artist Two"}
    # ]
    return jsonify(MeloMatch.RecommendationDF)

if __name__ == '__main__':
    views.run(debug=True)
