import sys
import traceback
from collections import OrderedDict

import numpy as np
import torch

old_stdout = sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import base64
import json
import pandas
import io
import os

BATCH_SIZE=16

def pandas_train_test_split(df, frac=0.2):
    test = df.sample(frac=frac, axis=0)
    train = df.drop(index=test.index)
    return train, test

def get_loaders(param_df=None,target_label='label'):
    df_train,df_test=pandas_train_test_split(param_df)

    train_features_tensor = torch.tensor(df_train.drop(target_label, axis=1).values.astype(np.float32))
    train_output_tensor = torch.LongTensor(df_train[target_label].values.astype(np.longlong))
    train_dataset = torch.utils.data.TensorDataset(train_features_tensor,train_output_tensor)

    test_features_tensor = torch.tensor(df_train.drop(target_label, axis=1).values.astype(np.float32))
    test_output_tensor = torch.LongTensor(df_train[target_label].values.astype(np.longlong))
    test_dataset = torch.utils.data.TensorDataset(test_features_tensor,test_output_tensor)

    result_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    result_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_features_tensor.size(dim=1),result_train_loader,result_test_loader

def train_epoch_emb(net, dataloader, lr=0.01, optimizer=None, loss_fn=torch.nn.CrossEntropyLoss(), epoch_size=None,
                    report_freq=200,device="cpu"):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    total_loss, acc, count, i = 0, 0, 0, 0
    for embeddings, labels in dataloader:
        optimizer.zero_grad()
        labels, embeddings = labels.to(device), embeddings.to(device)
        out = net(embeddings)
        loss = loss_fn(out, labels)  # cross_entropy(out,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)
        acc += (predicted == labels).sum()
        count += len(labels)
        i += 1
        if i % report_freq == 0:
            print(f"{count}: acc={acc.item() / count}",file=sys.__stderr__,flush=True)
        if epoch_size and count > epoch_size:
            break
    return total_loss.item() / count, acc.item() / count

# read configurations
buf=''
while True:
    msg=input()
    buf=buf+msg
    if "\t\t\t" in msg:
        config = json.loads(base64.b64decode(buf))
        buf=""
        break
    else:
        continue
save = config['save']


while True:
    msg=input()
    buf=buf+msg
    #read request
    if "\t\t\t" in msg:
        try:
            data = json.loads(base64.b64decode(buf))
        except Exception as e:
            print(e)
            if os.environ.get('DEBUG','0')=='1':
                print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
                raise e
            continue
        buf=""
    else:
        continue
    if "config" in data:
        new_config = data['config']
        if new_config.get('savePath') and new_config.get('saveName'):
            config['save']=os.path.join(new_config['savePath'], new_config['saveName'])
        if new_config.get('orient'):
            config['orient']=new_config['orient']
        if new_config.get('epochCount'):
            config['epochCount']=new_config['epochCount']
        save = config['save']
        sys.stdout = old_stdout
        print(json.dumps({"state":"parameters applied","config":config}), flush=True)
        sys.stdout=silent_stdout
        continue

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
        categor = pandas.Categorical(df['label'])
        id2label = dict(enumerate(categor.categories))
        label2id = {y: x for x, y in id2label.items()}
        df['label'] = pandas.DataFrame(categor.codes)
        input_size,train_loader,test_loader = get_loaders(df)
        df=None

        num_labels = len(id2label)
        hidden_size=int(input_size*1.5)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        new_net=torch.nn.Sequential(OrderedDict([
            ('dropout',torch.nn.Dropout(0.1)),
            ('hidden',torch.nn.Linear(input_size,hidden_size)),
            ('output',torch.nn.Linear(hidden_size,num_labels)),
        ])).to(device)

        for e in range(config['epochCount']):
            train_epoch_emb(new_net, train_loader, lr=2e-5,report_freq=10,device=device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            total_loss, acc, count, i = 0, 0, 0, 0
            for embeddings, labels in test_loader:
                labels, embeddings = labels.to(device), embeddings.to(device)
                out = new_net(embeddings)
                loss = loss_fn(out, labels)  # cross_entropy(out,labels)
                total_loss += loss
                _, predicted = torch.max(out, 1)
                acc += (predicted == labels).sum()
                count += len(labels)
                i += 1
            print(f'total_loss={total_loss.item() / count}, acc={acc.item() / count}',file=sys.__stderr__,flush=True)

        dir = os.path.dirname(save)
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

        torch.save(new_net, save+"/model.pickle")
        with open(save+"/id2label.json","wb") as f:
            f.write(json.dumps(id2label).encode())
        content=json.dumps({"state":"training completed"})
        sys.stdout = old_stdout
        print(base64.b64encode(content.encode()).decode('utf-8')+"\t\t\t\n",file=sys.__stdout__,flush=True)
        sys.stdout=silent_stdout
    except Exception as e:
        print(e)
        if os.environ.get('DEBUG','0')=='1':
            print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
            raise e
        continue