import pickle
from flask import Flask,request,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

RNDM_FRST_model=pickle.load(open('RNDM_FRST_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
    

@app.route('/pred_api_RNDM_FRST',methods=['POST'])
def pred_api_RNDM_FRST():
    data = request.json['data']

    # this "request" will actually help you to capture data(json/dict data), 
    # that is comming from the Postman

    print("\nrequest.json['data'] = ",data,"\n")
    required_data = [list(data.values())]
    print("\n2D data = ",required_data,"\n")

    Output = RNDM_FRST_model.predict(required_data)[0]

    return jsonify({'prediction': Output})

@app.route('/predict',methods=['POST'])
def predict():
    # when we submited data in "html" page 
    # how our python code(app.py)/model is retrive here that
    # means from help of request.form.values we get the data 
    # 
    # for retriving the data from "html page" we use 
    # below steps

    data=request.form.values() # here we will get all the values

    print("\nrequest.form.values() DATA : ",data,"\n")
    data1=[float(i) for i in data]
    print("\nfrom html page to we got data1 : ",data1,"\n")
    final_features = [np.array(data1)]

    # data = [float(i) for i in request.form.values()]
    # final_features = [np.array(data)]

    print("\nconverting to 2D as final_features (np.array(data1))  =",final_features,"\n")

    Output = RNDM_FRST_model.predict(final_features)[0]
    print("\nFROM HTML PAGE to 'DATA' OUTPUT PREDICTED VALUE(Output) = ",Output,"\n")
    
     # here prediction_text(place holder) is inside home.html file that will map the Output 
     # with home.html file
    return render_template("home.html",prediction_text="Airfoil pressure is {}".format(Output))

if __name__ == "__main__":
    app.run(debug=True)


# this bellow code is for if you give batch of inputs you get batch of outputs

# app = Flask(__name__)

# # Load your trained model
# RNDM_FRST_model = pickle.load(open('RNDM_FRST_model.pkl', 'rb'))

# @app.route('/pred_api_RNDM_FRST', methods=['POST'])
# def pred_api_RNDM_FRST():
#     data = request.json['data']
    
#     # Convert input data to DataFrame
#     df = pd.DataFrame(data)  # Handles multiple records

#     print("Received DataFrame:")
#     print(df)

#     # Predict for all rows
#     predictions = RNDM_FRST_model.predict(df).tolist()  # Convert to list for JSON

#     return jsonify({'predictions': predictions})

# if __name__ == "__main__":
#     app.run(debug=True)