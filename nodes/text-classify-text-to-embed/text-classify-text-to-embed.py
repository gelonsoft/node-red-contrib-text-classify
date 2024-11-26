import sys

from langchain_huggingface import HuggingFaceEmbeddings

old_stdout = sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import base64
import json
import pandas
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')
from sklw import SKLW
from preprocess_text import preprocess_text

REG_DETECTORS = ['text-classify-text-to-embed']

# read configurations
buf = ''
while True:
    msg = input()
    buf = buf + msg
    if "\t\t\t" in msg:
        config = json.loads(base64.b64decode(buf))
        buf = ""
        break
    else:
        continue
save = config['save']
hf_embeddings = None

while True:
    msg = input()
    buf = buf + msg
    # read request
    if "\t\t\t" in msg:
        try:
            data = json.loads(base64.b64decode(buf))
        except Exception as e:
            print(e)
            continue
        buf = ""
    else:
        continue
    if "config" in data:
        new_config = data['config']
        if new_config.get('savePath') and new_config.get('saveName'):
            config['save'] = os.path.join(new_config['savePath'], new_config['saveName'])
        if new_config.get('orient'):
            config['orient'] = new_config['orient']
        if new_config.get('modelPathOrName'):
            config['modelPathOrName'] = new_config['modelPathOrName']
        save = config['save']
        sys.stdout = old_stdout
        print(json.dumps({"state": "parameters applied", "config": config}), flush=True)
        sys.stdout = silent_stdout
        continue

    # Lazy load
    if hf_embeddings is None:
        hf_embeddings = HuggingFaceEmbeddings(model_name=config['modelNameOrPath'])

    #read request
    if 'file' in data:
        try:
            df = pandas.read_csv(data['file'])
        except Exception as e:
            print(e)
            continue
    else:
        try:
            #load data from request
            df = pandas.DataFrame.from_dict(data['data'], orient=config['orient'])
        except Exception as e:
            print(e)
            continue

    try:
        texts = list(df['text'].apply(preprocess_text).str.strip())
        if len(texts)<1:
            content=json.dumps({"data":{}})
            sys.stdout = old_stdout
            print(base64.b64encode(content.encode()).decode('utf-8') + "\t\t\t\n", flush=True)
            sys.stdout = silent_stdout
            continue
        embeddings = [ar for ar in hf_embeddings.embed_documents(texts)]
        emb_size=len(embeddings[0])
        content = json.dumps(dict({'x'+str(i):[v[i] for v in embeddings] for i in range(emb_size)}))
        sys.stdout = old_stdout
        print(base64.b64encode(content.encode()).decode('utf-8') + "\t\t\t\n", flush=True)
        sys.stdout = silent_stdout
    except Exception as e:
        print(e, file=sys.__stderr__, flush=True)
