# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('student_mark_predictor.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['study_hours']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    
    if output<49:
        res_val = "**and Fail **"
    else:
        res_val = "and Pass"
        
    return render_template('index.html', prediction_text='Your predicted Mark is  {}\n{}'.format(output,res_val))

if __name__ == "__main__":
    app.run()
