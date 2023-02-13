import pathlib
import json
from fastapi import FastAPI
from typing import Optional
import tensorflow as tf
import numpy as np
from keras.utils import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

class NumpyEncoder(json.JSONEncoder):
    ''' Special json encoder for numpy types'''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


app = FastAPI()

BASE_DIR = pathlib.Path(__file__).resolve().parent


MODEL_DIR = BASE_DIR.parent / "models"
SMS_SPAM_MODEL_DIR = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_MODEL_DIR / "spam-model.h5" 
TOKENIZER_PATH = SMS_SPAM_MODEL_DIR / "spam-classifer-tokenizer.json" 
METADATA_PATH = SMS_SPAM_MODEL_DIR / "spam-classifer-metadata.json" 


AI_MODEL = None
AI_TOKENIZER = None
MODEL_METADATA = {}
LEGEND_INVERTED = {}

@app.on_event("startup")
def on_startup():
    global AI_MODEL, AI_TOKENIZER, MODEL_METADATA, LEGEND_INVERTED, labels_legend_inverted
    # load my model
    if MODEL_PATH.exists():
        AI_MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if TOKENIZER_PATH.exists():
        t_json = TOKENIZER_PATH.read_text()
        AI_TOKENIZER = tokenizer_from_json(t_json)
        print(AI_TOKENIZER)

    if METADATA_PATH.exists():
        MODEL_METADATA = json.loads(METADATA_PATH.read_text())
        labels_legend_inverted = MODEL_METADATA['labels_legend_inverted']
        print(MODEL_METADATA)

def predict(query:str):
    # sequences
    # pad_sequence
    # model.predict
    # convert to labels
    sequences = AI_TOKENIZER.texts_to_sequences([query])
    maxlen = MODEL_METADATA.get('max_sequence') or 280
    x_input = pad_sequences(sequences, maxlen=maxlen)
    preds_array = AI_MODEL.predict(x_input) # array of preds
    
    preds = preds_array[0]
    top_idx_val = np.argmax(preds)
    top_pred = {
        "label": labels_legend_inverted[str(top_idx_val)],"confidence": preds[top_idx_val]
    } 
    
    labled_preds = [{"label": labels_legend_inverted[str(i)],"confidence": x} for i, x in enumerate(list(preds))]
    print(labled_preds)
    return json.loads(json.dumps({"top": top_pred, "predictions": labled_preds}, cls=NumpyEncoder))
    

@app.get("/") # /?q=this is awesome
def read_index(q:Optional[str] = None):
    global AI_MODEL, MODEL_METADATA, labels_legend_inverted
    query = q or "hello world"
    preds_dict = predict(query)
    print(AI_MODEL)
    return {
        "query": query,
        "result": preds_dict
    }