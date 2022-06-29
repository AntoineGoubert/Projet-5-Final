import tensorflow
import tensorflow_hub as hub
import mlinsights.mlmodel
import pickle
import streamlit as st

model = pickle.load(open('USE.pkl', 'rb'))
embedmodel = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
tsne=pickle.load(open("tsne.pkl",'rb'))

alldata = st.text_input("Texte")

features=tsne.transform(embedmodel([alldata]))
pred = model.predict(features)

st.write(pred[0])