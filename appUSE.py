from flask import Flask, render_template, request
import tensorflow_hub as hub
import pickle
import os
from pml import app

port = int(os.getenv('PORT'))

model = pickle.load(open('USE.pkl', 'rb'))
embedmodel = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
tsne=pickle.load(open("tsne.pkl",'rb'))

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def new():
    return render_template('new.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    data1 = str(request.form['a'])
    data2 = str(request.form['b'])
    alldata = data1 + ' ' + data2

    features=tsne.transform(embedmodel([alldata]))
    pred = model.predict(features)

    def statement():
        return('The most appropriate tag for your entry is ' + pred)

    return render_template('new.html', statement=statement())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)