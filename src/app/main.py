# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
import os
import pickle

#criar app
app = Flask(__name__)

app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

#movies
movies = pd.read_csv("../../data/raw/movies.csv")
# labeled movies
labeled_movies = pd.read_csv("data/raw/labeled_movies.csv")

#coluna de dados
colunas = ['nota_media', 'Drama', 'Documentary', 'Romance', 'Action',
       'Adventure', 'IMAX', 'Horror', 'War', 'Musical', 'Sci-Fi', 'Fantasy',
       'Animation', 'Film-Noir', 'Crime', 'Mystery', 'Children', 'Comedy',
       'Western', 'Thriller']

#colunas = ['id','filmeId']


def load_model( file_name = 'decision_tree.pkl'):
    return pickle.load(open(file_name, "rb"))

modelo = load_model('../../models/decision_tree.pkl')

@app.route('/reco/<id>')
def show_id(id):
    return f'Recebendo dados \n ID: {id}'

@app.route('/reco/', methods=['POST'])
@basic_auth.required
def get_recomendations():
   
    dados = request.get_json()
    playload = np.array([dados[col] for col in colunas])
    
    playload = playload.reshape(1, -1)
    cluster = np.float(modelo.predict(playload))
    cluster = int(cluster)
    return get_recomendations(cluster)
    
def get_recomendations(cluster):
    movie_group = labeled_movies.loc[labeled_movies['class'] == cluster]

    selected_movies = movie_group.sample(n = 3)

    recomendations = "Os filme que recomendamos para você são: \n"

    for index, row in selected_movies.iterrows():
        movie = movies.loc[movies['movieId'] == row['filmeId']].iloc[0]
        recomendations = recomendations + movie['title'] + " - " + movie['genres'] + "\n"

    return recomendations

@app.route('/')
def home():
    return 'API de predição de filmes'

app.run(debug=True, host='0.0.0.0')



