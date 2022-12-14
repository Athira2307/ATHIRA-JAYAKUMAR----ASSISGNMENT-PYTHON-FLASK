
#IMPORTING LIBRARIES
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index1.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run()