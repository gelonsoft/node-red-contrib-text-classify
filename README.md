# node-red-contrib-text-classify
This module for Node-RED contains a set of nodes which offer machine learning functionalities based on Berta model and Tensorflow.
Text classify predictions can be performed through the use of this package.

## Pre requisites
Be sure to have a working installation of [Node-RED](https://nodered.org/ "Node-RED").  
Install python and the following libraries:
* [Python](https://www.python.org/ "Python") 3.9.+ accessible by the command 'python' (on linux 'python3')
* [Numpy](http://www.numpy.org/ "Numpy")
* [Pandas](https://pandas.pydata.org/ "Pandas")
* [SciKit-Learn](http://scikit-learn.org "SciKit-Learn")
* [PyTorch](http://scikit-learn.org "Torch(PyTorch)")
* Full pip install: pip install scikit-learn evaluate transformers[torch] datasets nlp pandas nltk langchain_huggingface
* Run the following in your python after installation:
  * import nltk
  * nltk.download('stopwords')
  * nltk.download('punkt_tab')
  * nltk.download('wordnet')

## Install
To install the latest version use the Menu - Manage palette option and search for node-red-contrib-automl, or run the following command in your Node-RED user directory (typically ~/.node-red):

    npm i node-red-contrib-text-classify

## Usage
These flows create a dataset, train a model and then evaluate it. Models, after training, can be use in real scenarios to make predictions.
Models autotune hyperparameters with Optuna. 
Dataset must contain 'text' (input) and 'label' (target) columns.

Flows and test datasets are available in the 'test' folder. Make sure that the paths specified inside nodes' configurations are correct before trying to execute the program.  
**Tip:** you can run 'node-red' (or 'sudo node-red' if you are using linux) from the folder '.node-red/node-modules/node-red-contrib-text-classify' and the paths will be automatically correct.

This flow loads a training partition and trains a 'text-classify-trainer', saving the model locally.
![Training](https://i.imgur.com/oIDHwYu.png "Training")

This flow loads a test partition and evaluates a previously trained model.
![Evaluation](https://i.imgur.com/ufHBYLx.png "Evaluation")

You can use text classification model from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending "Hugging Face")

Example flows available here:
```json
[
  {
    "id": "cde349c1477e8ac6",
    "type": "tab",
    "label": "Example",
    "disabled": false,
    "info": "",
    "env": []
  },
  {
    "id": "caa9be34cfeb6e99",
    "type": "inject",
    "z": "cde349c1477e8ac6",
    "name": "Train data sample generator",
    "props": [
      {
        "p": "payload"
      }
    ],
    "repeat": "",
    "crontab": "",
    "once": false,
    "onceDelay": 0.1,
    "topic": "",
    "payload": "[ {\"text\":\"bla-bla\",\"label\":\"talk\"}, {\"text\":\"some message\",\"label\":\"talk\"}, {\"text\":\"I will kill you\",\"label\":\"warning\"}, {\"text\":\"fire at me\",\"label\":\"warning\"}, {\"text\":\"mine field\",\"label\":\"warning\"} ]",
    "payloadType": "json",
    "x": 360,
    "y": 140,
    "wires": [
      [
        "becc44c98f5e462d"
      ]
    ]
  },
  {
    "id": "610f718371104340",
    "type": "inject",
    "z": "cde349c1477e8ac6",
    "name": "Test data sample generator",
    "props": [
      {
        "p": "payload"
      }
    ],
    "repeat": "",
    "crontab": "",
    "once": false,
    "onceDelay": 0.1,
    "topic": "",
    "payload": "[ {\"text\":\"bla\"}, {\"text\":\"message\"}, {\"text\":\"kill\"}, {\"text\":\"fire\"}, {\"text\":\"mine\"} ]",
    "payloadType": "json",
    "x": 350,
    "y": 240,
    "wires": [
      [
        "9aed3d3835203404"
      ]
    ]
  },
  {
    "id": "9aed3d3835203404",
    "type": "text-classify-predictor",
    "z": "cde349c1477e8ac6",
    "name": "",
    "modelPath": "d:/a",
    "modelName": "somemodel",
    "orient": "records",
    "x": 640,
    "y": 240,
    "wires": [
      [
        "6ad8a705241d7924"
      ],
      [
        "3cbcdea3dcd22514"
      ]
    ]
  },
  {
    "id": "becc44c98f5e462d",
    "type": "text-classify-trainer",
    "z": "cde349c1477e8ac6",
    "name": "",
    "savePath": "d:/a",
    "saveName": "somemodel",
    "orient": "records",
    "x": 630,
    "y": 140,
    "wires": [
      [
        "a8a1ed5eafbf1964"
      ],
      [
        "0c6126c42bcbbc67"
      ]
    ]
  },
  {
    "id": "a8a1ed5eafbf1964",
    "type": "debug",
    "z": "cde349c1477e8ac6",
    "name": "good_messages",
    "active": true,
    "tosidebar": true,
    "console": false,
    "tostatus": false,
    "complete": "payload",
    "targetType": "msg",
    "statusVal": "",
    "statusType": "auto",
    "x": 900,
    "y": 100,
    "wires": []
  },
  {
    "id": "0c6126c42bcbbc67",
    "type": "debug",
    "z": "cde349c1477e8ac6",
    "name": "errors and warns",
    "active": true,
    "tosidebar": true,
    "console": false,
    "tostatus": false,
    "complete": "payload",
    "targetType": "msg",
    "statusVal": "",
    "statusType": "auto",
    "x": 910,
    "y": 160,
    "wires": []
  },
  {
    "id": "6ad8a705241d7924",
    "type": "debug",
    "z": "cde349c1477e8ac6",
    "name": "good_messages",
    "active": true,
    "tosidebar": true,
    "console": false,
    "tostatus": false,
    "complete": "payload",
    "targetType": "msg",
    "statusVal": "",
    "statusType": "auto",
    "x": 900,
    "y": 220,
    "wires": []
  },
  {
    "id": "3cbcdea3dcd22514",
    "type": "debug",
    "z": "cde349c1477e8ac6",
    "name": "errors and warns",
    "active": true,
    "tosidebar": true,
    "console": false,
    "tostatus": false,
    "complete": "payload",
    "targetType": "msg",
    "statusVal": "",
    "statusType": "auto",
    "x": 910,
    "y": 260,
    "wires": []
  }
]
```
## Thanks
Thanks to  Gabriele Maurina for awesome nodes - [node-red-contrib-machine-learning](https://github.com/GabrieleMaurina/node-red-contrib-machine-learning "node-red-contrib-machine-learning") 