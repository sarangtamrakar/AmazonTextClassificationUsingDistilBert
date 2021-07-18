from flask import Flask, Response, jsonify, request,render_template
from DistilBertModel import DistilBertModelClass
import logging

import json
import os

app = Flask(__name__)

logging.basicConfig(filename="flaskLogs.txt",level=logging.INFO,format='%(asctime)s  %(levelname)s %(name)s %(message)s')


clientObj = DistilBertModelClass()
Model = clientObj.LoadModel()
logging.info("Model Loaded")

@app.route("/",methods=["GET"])
def HomePage():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def PredictionRoute():
    if request.json is not None:
        text = request.json["text"]
        logging.info("Get the Text Data : "+str(text))
        try:
            prediction_value = clientObj.Predict(text, Model)
            logging.info("Prediction Value : "+str(prediction_value)+str(" Rating"))
            result1 = {
                "Prediction": str(prediction_value)+str(" Rating"),
                "result": str(text)
            }
        except Exception as e:
            logging.info("Model Failed")
            result1 = {
                "Prediction":"Model Failed"
            }

        return jsonify(result1)

if __name__ == "__main__":
    app.run()

