from flask import Flask, render_template, request
import tensorflow_hub as hub
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import sent_tokenize, word_tokenize

model = pickle.load(open('USE.pkl', 'rb'))
embedmodel = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
tsne=pickle.load(open("tsne.pkl",'rb'))
app = Flask(__name__)

def tokenizer_fct(sentence):
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ').replace('<', ' ').replace('>', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

def lower_start_fct(list_words):
    lw = [w.lower() for w in list_words if (not w.startswith("@"))
                                       and (not w.startswith("http"))]
    return lw

def transform_dl_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    lw = lower_start_fct(word_tokens)
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

@app.route('/', methods=['POST', 'GET'])
def new():
    return render_template('new.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    data1 = str(request.form['a'])
    data2 = str(request.form['b'])
    alldata = data1 + ' ' + data2

    features=tsne.transform(embedmodel([transform_dl_fct(alldata)]))
    pred = model.predict(features)

    def statement():
        return('The most appropriate tag for your entry is ' + pred)

    return render_template('new.html', statement=statement())


if __name__ == '__main__':
    app.run()