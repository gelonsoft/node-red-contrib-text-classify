import io
import sys

import numpy as np
import torch

old_stdout = sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import base64
import json
import pandas
from collections import OrderedDict #Do not remove!
import os

if os.environ.get('DISABLE_SSL_VERIFY', "0") == "1":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def pretty_output(some):
    result = []
    for a in some:
        row = {}
        for r in a:
            row[r['label']] = r['score'] * 100
        row = dict(sorted(row.items(), key=lambda item: item[1], reverse=True)[:5])
        result.append(row)
    return result


def load_model(save_path):
    _=OrderedDict() #Do not remove!
    lnet = torch.load(save_path + "/my_model.pickle", weights_only=False).to(device)
    with open(save_path + "/id2label.json", "rb") as f:
        lid2label = dict(json.load(f))
    lid2label = {int(k): v for k, v in lid2label.items()}
    return lid2label, lnet


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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = None
id2label = None
lastupdate = 0

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
        save = config['save']
        sys.stdout = old_stdout
        print(json.dumps({"state": "parameters applied", "config": config}), flush=True)
        sys.stdout = silent_stdout
        continue

    # read request
    if 'file' in data:
        try:
            df = pandas.read_csv(data['file'])
        except Exception as e:
            print(e)
            continue
    else:
        try:
            # load data from request
            df = pandas.read_json(io.StringIO(json.dumps(data).encode(errors='ignore').decode(encoding='utf-8',errors='ignore')), orient=config['orient'])
        except Exception as e:
            print(e)
            continue

    try:
        # Lazy load
        if net is None:
            id2label, net = load_model(save)
            lastupdate = os.stat(save + "/id2label.json").st_mtime
        else:
            modified = os.stat(save + "/config.json").st_mtime
            if modified > lastupdate:
                id2label, net = load_model(save)
                lastupdate = modified

        test_features_tensor = torch.tensor(df.values.astype(np.float32))
        with torch.no_grad():
            logits = net(test_features_tensor.to(device))

        outputs = logits.float().numpy()
        scores = softmax(outputs)
        dict_scores = [[
            {"label": id2label[i], "score": score.item()} for i, score in enumerate(scores_arr)
        ] for scores_arr in scores]
        dict_scores = [sorted(score, key=lambda x: x["score"], reverse=True) for score in dict_scores]

        content = json.dumps({"predict": pretty_output(dict_scores)})
        sys.stdout = old_stdout
        print(base64.b64encode(content.encode()).decode('utf-8') + "\t\t\t\n", flush=True)
        sys.stdout = silent_stdout
    except Exception as e:
        print(e, file=sys.__stderr__, flush=True)
