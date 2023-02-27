# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:37:55 2023

@author: Pariya
"""

from flask import Flask, request, jsonify, render_template
import joblib
from pyforest import *
import numpy as np

#naming our app as app
app= Flask(__name__)
#loading the pickle file for creating the web app
model= joblib.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
#specifying our parameters as data type float
    int_features= [float(x) for x in request.form.values()]
    final_features= [np.array(int_features)]
    prediction= model.predict(final_features)
    output= round(prediction[0], 2)
    return render_template("index.html", prediction_text= "flower is {}".format(output))
#running the flask app
if __name__== "__main__":
    app.run(debug=True)
