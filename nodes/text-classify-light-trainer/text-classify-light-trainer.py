import sys

old_stdout = sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import traceback
from collections import OrderedDict
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import base64
import json
import pandas
import io
import os

BATCH_SIZE = 16

import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        lloss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return lloss.mean()
        elif self.reduction == 'sum':
            return lloss.sum()
        return lloss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        lloss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return lloss.mean()
        elif self.reduction == 'sum':
            return lloss.sum()
        return lloss


def pandas_train_test_split(df, frac=0.2, target_label='label'):
    return train_test_split(df, test_size=frac, stratify=df[target_label])


def get_loaders(param_df=None, target_label='label'):
    maxx = max(param_df[target_label].value_counts().to_dict().keys())
    # print(maxx)
    df_train, df_test = pandas_train_test_split(param_df)
    x = dict(sorted(df_train[target_label].value_counts().to_dict().items()))
    x = {k: x[k] if k in x else sys.float_info.epsilon for k in range(maxx + 1)}
    # print(x)
    x = list(x.values())
    ma = max(x) + sys.float_info.epsilon
    alpha_train = torch.FloatTensor([v / ma for v in x])
    x = dict(sorted(df_test[target_label].value_counts().to_dict().items()))
    x = {k: x[k] if k in x else sys.float_info.epsilon for k in range(maxx + 1)}
    # print(x)
    x = list(x.values())
    ma = max(x) + sys.float_info.epsilon
    alpha_test = torch.FloatTensor([v / ma for v in x])

    train_features_tensor = torch.tensor(df_train.drop(target_label, axis=1).values.astype(np.float32))
    train_output_tensor = torch.LongTensor(df_train[target_label].values.astype(np.longlong))
    train_dataset = torch.utils.data.TensorDataset(train_features_tensor, train_output_tensor)

    test_features_tensor = torch.tensor(df_train.drop(target_label, axis=1).values.astype(np.float32))
    test_output_tensor = torch.LongTensor(df_train[target_label].values.astype(np.longlong))
    test_dataset = torch.utils.data.TensorDataset(test_features_tensor, test_output_tensor)

    result_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    result_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_features_tensor.size(dim=1), result_train_loader, result_test_loader, alpha_train, alpha_test


def train_epoch_emb(net, dataloader, lr=0.01, optimizer=None, loss_fn=torch.nn.CrossEntropyLoss(), epoch_size=None,
                    report_freq=200, device="cpu"):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    # loss_fn = loss_fn.to(device)
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
            print(f"{count}: acc={acc.item() / count}", file=sys.__stderr__, flush=True)
        if epoch_size and count > epoch_size:
            break
    return total_loss.item() / count, acc.item() / count


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

while True:
    msg = input()
    buf = buf + msg
    # read request
    if "\t\t\t" in msg:
        try:
            data = json.loads(base64.b64decode(buf))
        except Exception as e:
            print(e)
            if os.environ.get('DEBUG', '0') == '1':
                print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
                raise e
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
        if new_config.get('epochCount'):
            config['epochCount'] = new_config['epochCount']
        save = config['save']
        sys.stdout = old_stdout
        print(json.dumps({"state": "parameters applied", "config": config}), flush=True)
        sys.stdout = silent_stdout
        continue

    if 'file' in data:
        try:
            df = pandas.read_csv(data['file'])
        except Exception as e:
            print(e)
            if os.environ.get('DEBUG', '0') == '1':
                print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
                raise e
            continue
    else:
        try:
            # load data from request
            df = pandas.read_json(
                io.StringIO(json.dumps(data).encode(errors='ignore').decode(encoding='utf-8', errors='ignore')),
                orient=config['orient'])
        except Exception as e:
            print(e)
            if os.environ.get('DEBUG', '0') == '1':
                print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
                raise e
            continue

    try:
        df = df[df.groupby('label').label.transform(len) > 1]
        df = df.reset_index(drop=True)
        categor = pandas.Categorical(df['label'])
        id2label = dict(enumerate(categor.categories))
        label2id = {y: x for x, y in id2label.items()}
        df['label'] = pandas.DataFrame(categor.codes)
        input_size, train_loader, test_loader, alpha_train, alpha_test = get_loaders(df)
        df = None

        num_labels = len(id2label)
        hidden_size = int(input_size * 1.5)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        new_net = torch.nn.Sequential(OrderedDict([
            ('dropout', torch.nn.Dropout(0.1)),
            ('hidden', torch.nn.Linear(input_size, hidden_size)),
            ('output', torch.nn.Linear(hidden_size, num_labels)),
        ])).to(device)

        criterion = FocalLoss(gamma=2, alpha=alpha_train, task_type='multi-class', num_classes=num_labels)
        for e in range(config['epochCount']):
            train_epoch_emb(new_net, train_loader, lr=2e-5, report_freq=10, device=device, loss_fn=criterion)
        criterion = FocalLoss(gamma=2, alpha=alpha_test, task_type='multi-class', num_classes=num_labels)
        with torch.no_grad():
            total_loss, acc, count, i = 0, 0, 0, 0
            for embeddings, labels in test_loader:
                labels, embeddings = labels.to(device), embeddings.to(device)
                out = new_net(embeddings)
                loss = criterion(out, labels)  # cross_entropy(out,labels)
                total_loss += loss
                _, predicted = torch.max(out, 1)
                acc += (predicted == labels).sum()
                count += len(labels)
                i += 1
            print(f'total_loss={total_loss.item() / count}, acc={acc.item() / count}', file=sys.__stderr__, flush=True)

        if not os.path.isdir(save):
            os.makedirs(save, exist_ok=True)

        torch.save(new_net, save + "/model.pickle")
        with open(save + "/id2label.json", "wb") as f:
            f.write(json.dumps(id2label).encode())
        content = json.dumps({"state": "training completed"})
        sys.stdout = old_stdout
        print(base64.b64encode(content.encode()).decode('utf-8') + "\t\t\t\n", file=sys.__stdout__, flush=True)
        sys.stdout = silent_stdout
    except Exception as e:
        print(e)
        if os.environ.get('DEBUG', '0') == '1':
            print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
            raise e
        continue
