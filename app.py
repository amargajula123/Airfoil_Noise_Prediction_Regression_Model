import pickle
from flask import Flask,request,jsonify
import numpy as np
import pandas as pd

app=Flask(__name__)

RNDM_FRST_model=pickle.load(open('RNDM_FRST_model.pkl','rb'))

@app.route('/pred_api_RNDM_FRST',methods=['POST'])
def pred_api_RNDM_FRST():
    data = request.json['data']
    # this "request" will actually help you to capture data, 
    # that is comming from the Postman

    print("request.json['data'] = ",data)
    required_data = [list(data.values())]
    print("2D data = ",required_data)

    Output = RNDM_FRST_model.predict(required_data)[0]

    return jsonify({'prediction': Output})

if __name__ == "__main__":
    app.run(debug=True)
