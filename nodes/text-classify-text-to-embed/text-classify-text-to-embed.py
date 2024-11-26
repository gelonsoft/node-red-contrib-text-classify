import io
import sys
import traceback

from langchain_huggingface import HuggingFaceEmbeddings

old_stdout = sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import base64
import json
import pandas
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')
from sklw import SKLW
from preprocess_text import preprocess_text

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
            if os.environ.get('DEBUG','0')=='1':
                print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
                raise e
            continue
        buf = ""
    else:
        continue
    if "config" in data:
        new_config = data['config']
        if new_config.get('orient'):
            config['orient'] = new_config['orient']
        if new_config.get('modelPathOrName'):
            config['modelPathOrName'] = new_config['modelPathOrName']
            if hf_embeddings is None:
                hf_embeddings = HuggingFaceEmbeddings(model_name=config['modelPathOrName'])
        sys.stdout = old_stdout
        print(json.dumps({"state": "parameters applied", "config": config}), flush=True)
        sys.stdout = silent_stdout
        continue

    # Lazy load
    if hf_embeddings is None:
        hf_embeddings = HuggingFaceEmbeddings(model_name=config['modelPathOrName'])

    #read request
    if 'file' in data:
        try:
            df = pandas.read_csv(data['file'])
        except Exception as e:
            print(e)
            if os.environ.get('DEBUG','0')=='1':
                print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
                raise e
            continue
    else:
        try:
            #load data from request
            df = pandas.read_json(io.StringIO(json.dumps(data).encode(errors='ignore').decode(encoding='utf-8',errors='ignore')), orient=config['orient'])
        except Exception as e:
            print(e)
            if os.environ.get('DEBUG','0')=='1':
                print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
                raise e
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
        df=df.drop(['text'],axis=1)
        df=df.assign(**dict({'x'+str(i):[v[i] for v in embeddings] for i in range(emb_size)}))
        #content = json.dumps(dict({'x'+str(i):[v[i] for v in embeddings] for i in range(emb_size)}))
        content = json.dumps(df.to_dict(orient='list'))
        sys.stdout = old_stdout
        print(base64.b64encode(content.encode()).decode('utf-8') + "\t\t\t\n", flush=True)
        sys.stdout = silent_stdout
    except Exception as e:
        print(e, file=sys.__stderr__, flush=True)
        if os.environ.get('DEBUG','0')=='1':
            print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
            raise e

