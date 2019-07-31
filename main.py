from flask import Flask, request, jsonify,session, render_template,redirect, url_for
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import warnings
import heapq
import random
import requests
import json
import ast
import re
import os
from flask_cors import CORS
warnings.filterwarnings('ignore')

app = Flask(__name__)
cors = CORS(app, resources={r"/getIntent/*": {"origins": "*"}})
abs_path = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(abs_path,"dataBank/tokenizer_GRU.pickle")
model_path = os.path.join(abs_path,"dataBank/lang_model_GRU.h5")
try:
    with open(tokenizer_path,"rb") as file:
        tokenizer = pickle.load(file)
except:
    print("cannot load tokenizer")

try:
    model = load_model(model_path)
except:
    print("cannot load model")


class Preprocessore:
  
  def __init__(self):
    pass
  
  def preprocess_df(self,df=None):
    preprocessed_texts = []
    label = []
    for i in range(len(df)):
        s = re.sub(r"[?,']", "", df.iloc[i].text)
        if len(s)>0:
            preprocessed_texts.append(s)
            label.append(label_dict[df.iloc[i].label])
    new_df = pd.DataFrame({"text":preprocessed_texts,"label":label})
    new_df = shuffle(new_df)
    new_df = shuffle(new_df)
    return new_df
  
  def preprocess_list(self,seq_list=None,w2v=False):
    final_texts = []
    for elem in seq_list:
      s = re.sub(r"[?,']", "", elem)
      if len(s)>0:
        if w2v:
          final_texts.append(s.split(" "))
        else:
          final_texts.append(s)
    return final_texts      


@app.route("/getIntent",methods=["GET","POST"])
def getIntent():
    
    label_dict = {'annual-fee':0,'eligibility':1,'facility':2,'interest-rate':3,'mobile-recharge':4,'required-documents':5}
    idx2cls = {0:'annual-fee',1:'eligibility',2:'facility',3:'interest-rate',4:'Unknown',5:'required-documents'}
    
    if request.method == "GET":
        payloads = {"intent":"","confidence":{'annual-fee':0,'eligibility':1,'facility':2,'interest-rate':3,'mobile-recharge':4,'required-documents':5}}
        user = request.args.get('data')
        preprocessore = Preprocessore()
        s = []
        s.append(user)
        s = preprocessore.preprocess_list(s)
        
        d = tokenizer.texts_to_sequences(s)
        d = pad_sequences(d,100)
        pred = model.predict(d)
        cl = idx2cls[np.argmax(pred[0,:])]
        payloads["intent"] = cl
        rr = {}
        for ind, keys in enumerate(['annual-fee','eligibility','facility','interest-rate','Unknown','required-documents']):
            rr[keys] = str(pred[0][ind])

        payloads["confidence"] = rr
        payloads["language"] = "en"
  

        return json.dumps(payloads)

    else:
        return json.dumps(['This method is not allowed'])
        

if __name__=="__main__":
    app.run(debug=True)