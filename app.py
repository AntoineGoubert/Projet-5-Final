from flask import Flask, render_template, request
import tensorflow_hub as hub
import pickle
import streamlit as st

model = pickle.load(open('USE.pkl', 'rb'))
embedmodel = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
tsne=pickle.load(open("tsne.pkl",'rb'))

data1 = str(request.form['a'])
data2 = str(request.form['b'])
alldata = data1 + ' ' + data2

features=tsne.transform(embedmodel([alldata]))
pred = model.predict(features)

st.write(pred)