# node-red-contrib-text-classify
This module for Node-RED contains a set of nodes which offer machine learning functionalities based on Berta model and Tensorflow.
Text classify predictions can be performed through the use of this package.

## Pre requisites
Be sure to have a working installation of [Node-RED](https://nodered.org/ "Node-RED").  
Install python and the following libraries:
* [Python](https://www.python.org/ "Python") 3.9.0 ) accessible by the command 'python' (on linux 'python3')
* [Numpy](http://www.numpy.org/ "Numpy")
* [Pandas](https://pandas.pydata.org/ "Pandas")
* [SciKit-Learn](http://scikit-learn.org "SciKit-Learn")
* Full pip install: pip install numpy==1.26.4 nltk pandas==2.2.3 tf-models-official==2.10.0 tensorflow==2.10.1 tensorflow_text==2.10.0 tensorflow_addons==0.20.0 tensorflow_hub 
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

Flows and test datasets are available in the 'test' folder. Make sure that the paths specified inside nodes' configurations are correct before trying to execute the program.  
**Tip:** you can run 'node-red' (or 'sudo node-red' if you are using linux) from the folder '.node-red/node-modules/node-red-contrib-text-classify' and the paths will be automatically correct.

This flow loads a training partition and trains a 'text-classify-trainer', saving the model locally.
![Training](https://imgur.com/OBuB6LZ.png "Training")

This flow loads a test partition and evaluates a previously trained model.
![Evaluation](https://imgur.com/tTk34y5.png "Evaluation")

By default it uses Bert Encoder and Bert Preprocessor models, but you can download alternative from https://www.kaggle.com/models/tensorflow/bert/tensorFlow2/ , unpack .tar.gz to any dirs and specify new model location by setting the following environment variables:
* TC_BERT_PREPROCESSOR_DIR (by default uses bert-tensorflow2-en-uncased-preprocess-v3 model from ./bert/preprocessor subdirectory )
* TC_BERT_ENCODER_DIR (by default uses bert-tensorflow2-bert-en-uncased-l-4-h-128-a-2-v2 model from ./bert/encoder subdirectory)

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
    "payload": "[ {\"x\":\"bla-bla\",\"y\":\"talk\"}, {\"x\":\"some message\",\"y\":\"talk\"}, {\"x\":\"I will kill you\",\"y\":\"warning\"}, {\"x\":\"fire at me\",\"y\":\"warning\"}, {\"x\":\"mine field\",\"y\":\"warning\"} ]",
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
    "payload": "[ {\"x\":\"bla\"}, {\"x\":\"message\"}, {\"x\":\"kill\"}, {\"x\":\"fire\"}, {\"x\":\"mine\"} ]",
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