import tensorflow
import tensorflow_hub as hub
import mlinsights.mlmodel
import pickle
import streamlit as st
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = pickle.load(open('USE.pkl', 'rb'))
embedmodel = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
tsne=pickle.load(open("tsne.pkl",'rb'))

data1=st.text_input("Titre")
data2=st.text_input("Texte")
alldata = data1 + ' ' + data2

features=tsne.transform(embedmodel([alldata]))
pred = model.predict(features)

st.write(pred[0])