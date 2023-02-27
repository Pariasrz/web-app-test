# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:37:55 2023

@author: Pariya
"""

#importing the necessary libraries for deployment
import pickle
from flask import Flask, request, jsonify, render_template
from pyforest import *
import numpy as np
#naming our app as app
app= Flask(__name__)
#loading the pickle file for creating the web app
model= pickle.load(open('/mnt/c/Users/Pariya/Desktop/Program/finalized_model.sav', 'rb'))
#defining the different pages of html and specifying the features required to be filled in the ht

@app.route("/")
def home():
    return render_template("index.html")

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