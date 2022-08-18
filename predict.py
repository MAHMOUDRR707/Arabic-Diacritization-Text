import numpy as np
import pickle as pkl
import pickle as pkl
from map_data import map_data
from remove_diacritics import remove_diacritics
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
import numpy as np
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Arabic Diacritization Model API",
    description="A simple API that use NLP model to predict Arabic Diacritization ",
    version="0.1",
)
model =  load_model("output.h5")

with open('ARABIC_LETTERS_LIST.pickle', 'rb') as file:
    ARABIC_LETTERS_LIST = pkl.load(file)
with open('DIACRITICS_LIST.pickle', 'rb') as file:
    DIACRITICS_LIST = pkl.load(file)
with open('RNN_SMALL_CHARACTERS_MAPPING.pickle', 'rb') as file:
        CHARACTERS_MAPPING = pkl.load(file)
with open('RNN_CLASSES_MAPPING.pickle', 'rb') as file:
    CLASSES_MAPPING = pkl.load(file)
with open('RNN_REV_CLASSES_MAPPING.pickle', 'rb') as file:
    REV_CLASSES_MAPPING = pkl.load(file)






@app.get("/predict-review")
def predict(line):
    X, _ = map_data([line])
    predictions = model.predict(X).squeeze()
    predictions = predictions[1:]

    output = ''
    for char, prediction in zip(remove_diacritics(line), predictions):
        output += char

        if char not in ARABIC_LETTERS_LIST:
            continue

        if '<' in REV_CLASSES_MAPPING[np.argmax(prediction)]:
            continue

        output += REV_CLASSES_MAPPING[np.argmax(prediction)]

    return {"output":output}

