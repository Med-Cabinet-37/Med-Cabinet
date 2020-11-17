
from flask import Flask, render_template, request
import numpy as np
import pandas as pd 
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.losses import sparse_categorical_crossentropy

df = pd.read_csv('data/cannabis.csv')
y = pd.read_csv('data/lemmas.csv')['Strain']

best_model = joblib.load('neural-net.joblib')

def create_app():

    app = Flask(__name__)

    @app.route('/')
    def root():
        return render_template(
            'base.html',
            title='Home',
        )

    @app.route('/find', methods=['POST'])
    def find():
        user_text = request.values["user_text"]
        message = predict(user_text)
        return render_template(
            'base.html',
            message=message
        )

    return app


def clean_example(ex: str) -> str:
    tokens = [
    token.lemma_ for token in nlp(ex)
        if not token.is_stop
        if not token.is_punct
        if not token.is_space
    ]
    return " ".join(tokens)

def predict(ex: str) -> pd.Series:
    '''Takes in user's input and returns a prediction'''
    cleaned = clean_example(ex)
    as_vector = nlp(cleaned).vector.reshape(1, -1)
    prediction = np.argmax(
        best_model.predict(as_vector),
        axis=-1
    )[0]
    strain = y.iloc[prediction]
    return df[df['Strain'] == strain].iloc[0]