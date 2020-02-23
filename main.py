from flask import Flask, render_template, request, session,redirect
from joblib import dump, load
import numpy as np
model=load('houseratepred.joblib')
app= Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=['POST'])
def predict():
    features=[float(value) for value in request.form.values()]
    final_features=[np.array(features)]
    prediction=model.predict(final_features)
    return render_template("index.html",prediction_text="House price should be {} lakhs".format(prediction))
if __name__== '__main__':
    app.run(debug=True,use_reloader=False)
