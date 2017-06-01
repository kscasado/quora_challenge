from flask import Flask, jsonify, render_template
from model/model_creater import ModelCreater
from model/model_server import ModelServer

app = Flask('Semantic Similarity')


@app.route('/model/predict')
def make_prediction():
