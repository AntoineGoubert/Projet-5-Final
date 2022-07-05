import tensorflow
import tensorflow_hub as hub
import mlinsights.mlmodel
import pickle
import streamlit as st
import os


model = pickle.load(open("C:/Users/antoi/Dropbox/PC/Documents/GitHub/Projet-5-Final/USE.pkl", 'rb'))
embedmodel = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
tsne=pickle.load(open("C:/Users/antoi/Dropbox/PC/Documents/GitHub/Projet-5-Final/tsne.pkl",'rb'))

data1=st.text_input("Titre")
data2=st.text_input("Texte")
alldata = data1 + ' ' + data2

features=tsne.transform(embedmodel([alldata]))
pred = model.predict(features)

st.write(pred)