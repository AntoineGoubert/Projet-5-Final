import tensorflow
import tensorflow_hub as hub
import mlinsights.mlmodel
import pickle
import streamlit as st
import numpy as np

model = pickle.load(open("C:/Users/antoi/Dropbox/PC/Documents/GitHub/Projet-5-Final/USE.pkl", 'rb'))
embedmodel = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
tsne=pickle.load(open("C:/Users/antoi/Dropbox/PC/Documents/GitHub/Projet-5-Final/tsne.pkl",'rb'))
tags=np.load("C:/Users/antoi/Dropbox/PC/Documents/GitHub/Projet-5-Final/tags.npy")

st.title("StackOverflow tags prediction")

def predict_class():
    alldata = data1 + ' ' + data2
    features = tsne.transform(embedmodel([alldata]))
    pred = model.predict_proba(features)
    liste = []
    for k in range(len(np.argpartition(pred, -3)[0][-3:])):
        liste += [tags[k]]
        print(tags[k])
    st.write("The three most likely tags are, in decreasing probability order : ", liste)

data1=st.text_input("Title")
data2=st.text_input("Body")
if st.button("Predict"):
    predict_class()
