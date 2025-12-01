import pickle
from django.shortcuts import render
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
##Load the model
pickle_model = pickle.load(open("delivery_cost_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json('data')
    print(data)
    data_df = pd.DataFrame(data)
    output = pickle_model.predict(data_df)
    print(output[0])
    return jsonify({'prediction': list(output[0])})

@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = pd.DataFrame(np.array(data).reshape(1,-1))
    print(final_input)
    output = pickle_model.predict(final_input)[0]
    print(output)
    return render_template("home.html", prediction_text = "The predicted delivery cost is Rs. {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)

