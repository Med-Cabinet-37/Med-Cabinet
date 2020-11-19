
from flask import Flask, render_template, request, make_response, jsonify
import numpy as np
import pandas as pd 
import joblib
import spacy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.losses import sparse_categorical_crossentropy

df = pd.read_csv('data/cannabis_final.csv')

best_model = load_model('neural-network-final.h5')

nlp = spacy.load('model_final')

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
    prediction = best_model.predict(as_vector)
    strain = np.argmax(prediction, axis=-1)[0]
    return df.iloc[strain]