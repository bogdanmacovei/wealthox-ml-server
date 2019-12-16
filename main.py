from flask import Flask
from flask import jsonify, request, Response
from flask_cors import CORS, cross_origin

import json
import pickle

import pandas
import numpy as np

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn import preprocessing

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

import utils

# C:/Users/bogda/AppData/Local/Continuum/anaconda3/envs/ML/python.exe main.py

globalPath = 'model.csv'

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/houses-suggestion": {"origins": "https://localhost:44376"}})
cors = CORS(app, resources={r"/use=gradient-boosting-regressor": {"origins": "http://0.0.0.0"}})




@app.route("/train-gradient-boosting-regressor")
def trainGradientBoostingRegressor():
    _, X, y = utils.gettingData(path=globalPath, colsX=utils.getCols(), colsLabely=['Label'])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1)

    gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.001, max_features="sqrt", max_depth=10, loss='lad', verbose=1)
    gbr.fit(X_train, y_train)

    filename = 'gbr_model.sav'
    pickle.dump(gbr, open(filename, 'wb'))
    return jsonify(result="model trained")

@app.route("/test-gradient-boosting-regressor/<ind>")
def testGradientBoostingRegressor(ind):
    df, _, _ = utils.gettingData(path=globalPath, colsX=utils.getCols(), colsLabely=['Label'])

    filename = 'gbr_model.sav'
    gbr = pickle.load(open(filename, 'rb'))
    prediction = gbr.predict(df[utils.getCols()].values[int(ind)].reshape(1, -1))[0]

    return jsonify(result=str(prediction), real=str(df.values[int(ind), 35]))

@app.route("/use-gradient-boosting-regressor", methods=['POST'])
def useGradientBoostingRegressor():
    to_predict = []
    for col in utils.getCols():
        print(request.json.get(col))
        to_predict += [request.json.get(col)]
    to_predict = np.array(to_predict).reshape(1, -1)

    filename = 'gbr_model.sav'
    gbr = pickle.load(open(filename, 'rb'))
    prediction = gbr.predict(to_predict)[0]

    return jsonify(result=str(prediction))

@app.route("/houses-suggestion", methods=['POST'])
@cross_origin(origin='localhost:44376',headers=['Content- Type','Authorization'])
def housesSuggestion():
    to_parse = str(request.get_data())
    parsed = to_parse.split('&')
    to_predict = []
    for p in parsed:
        value = float(p.split('=')[1].split('\'')[0])
        to_predict += [value]
    
    to_predict = np.array([to_predict])
    _, X, _ = utils.gettingData(path=globalPath, colsX=utils.getFilters(), colsLabely=['Label'])
    
    kNN = NearestNeighbors(n_neighbors=10)
    kNN.fit(X)
    
    result = kNN.kneighbors(to_predict, return_distance=False)

    return jsonify(result=pandas.Series(result[0]).to_json(orient='values'))

@app.route("/")
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True)