
from flask import Flask, render_template, request, make_response, jsonify
import numpy as np
import pandas as pd 
import joblib
import spacy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.losses import sparse_categorical_crossentropy

df = pd.read_csv('data/cannabis.csv')
y = pd.read_csv('data/lemmas.csv')['Strain']

#nn = joblib.load('nearest_neighbors.joblib')
best_model = load_model('neural-network.h5')

nlp = spacy.load('model')

def create_app():

    app = Flask(__name__)

    @app.route('/')
    def root():
        return render_template(
            'base.html',
            title='Home',
        )

    @app.route('/find', methods=['GET','POST'])
    def find():
        if request.method == 'POST':
            message = predict(request.form.to_dict()['user_text'])
            #message = "Testing..."
            return make_response(jsonify({"strain":str(message['Strain']),
                                        "flavor":str(message['Flavor']),
                                        "effects":str(message['Effects']),
                                        "description":str(message['Description'])
                                        }), 200)
        else:
            return "Return else statement"

    return app

# def clean_example(ex: str) -> str:
#     tokens = [
#         token.lemma_ for token in nlp(ex)
#             if not token.is_stop
#             if not token.is_punct
#             if not token.is_space
#     ]
#     return " ".join(tokens)
# def predict(desc: str) -> pd.DataFrame:
#     if len(desc) == 0:
#         return 'N/A'
#     vector = nlp(clean_example(desc)).vector.reshape(1, -1)
#     # returns the n = 5 nearest neighbors
#     n = 1
#     result = nn.kneighbors(vector, n)
#     return df.iloc[result[1][0]]

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
    if as_vector.shape[1] == 0:
        return "More information needed"
    else:
        prediction = np.argmax(
            best_model.predict(as_vector),
            axis=-1
        )[0]
        strain = y.iloc[prediction]
        return df[df['Strain'] == strain].iloc[0]