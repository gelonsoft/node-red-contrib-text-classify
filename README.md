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
Dataset must contain 'text' (input) and 'label' (target) columns.

Flows and test datasets are available in the 'test' folder. Make sure that the paths specified inside nodes' configurations are correct before trying to execute the program.  
**Tip:** you can run 'node-red' (or 'sudo node-red' if you are using linux) from the folder '.node-red/node-modules/node-red-contrib-text-classify' and the paths will be automatically correct.

This flow loads a training partition and trains a 'text-classify-trainer', saving the model locally.
![Training](https://i.imgur.com/oIDHwYu.png "Training")

This flow loads a test partition and evaluates a previously trained model.
![Evaluation](https://i.imgur.com/ufHBYLx.png "Evaluation")

You can also use text-classify-light* nodes as in example flow to split text embedding and classification tasks

You can use text classification model from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending "Hugging Face")

Example flows available here:
```json
[{"id":"cde349c1477e8ac6","type":"tab","label":"Example","disabled":false,"info":"","env":[]},{"id":"caa9be34cfeb6e99","type":"inject","z":"cde349c1477e8ac6","name":"Train data sample generator","props":[{"p":"payload"}],"repeat":"","crontab":"","once":false,"onceDelay":0.1,"topic":"","payload":"[ {\"text\":\"bla-bla\",\"label\":\"talk\"}, {\"text\":\"some message\",\"label\":\"talk\"}, {\"text\":\"I will kill you\",\"label\":\"warning\"}, {\"text\":\"fire at me\",\"label\":\"warning\"}, {\"text\":\"mine field\",\"label\":\"warning\"} ]","payloadType":"json","x":360,"y":140,"wires":[["becc44c98f5e462d"]]},{"id":"610f718371104340","type":"inject","z":"cde349c1477e8ac6","name":"Test data sample generator","props":[{"p":"payload"}],"repeat":"","crontab":"","once":false,"onceDelay":0.1,"topic":"","payload":"[ {\"text\":\"bla\"}, {\"text\":\"message\"}, {\"text\":\"kill\"}, {\"text\":\"fire\"}, {\"text\":\"mine\"} ]","payloadType":"json","x":350,"y":240,"wires":[["9aed3d3835203404"]]},{"id":"9aed3d3835203404","type":"text-classify-predictor","z":"cde349c1477e8ac6","name":"","modelPath":"/tmp","modelName":"somemodel","orient":"records","x":640,"y":240,"wires":[["6ad8a705241d7924"],["3cbcdea3dcd22514"]]},{"id":"becc44c98f5e462d","type":"text-classify-trainer","z":"cde349c1477e8ac6","name":"","savePath":"/tmp","saveName":"somemodel","tokenizerPathOrName":"cointegrated/LaBSE-en-ru","modelPathOrName":"cointegrated/LaBSE-en-ru","orient":"records","x":630,"y":140,"wires":[["a8a1ed5eafbf1964"],["0c6126c42bcbbc67"]]},{"id":"a8a1ed5eafbf1964","type":"debug","z":"cde349c1477e8ac6","name":"good_messages","active":true,"tosidebar":true,"console":false,"tostatus":false,"complete":"payload","targetType":"msg","statusVal":"","statusType":"auto","x":900,"y":100,"wires":[]},{"id":"0c6126c42bcbbc67","type":"debug","z":"cde349c1477e8ac6","name":"errors and warns","active":true,"tosidebar":true,"console":false,"tostatus":false,"complete":"payload","targetType":"msg","statusVal":"","statusType":"auto","x":910,"y":160,"wires":[]},{"id":"6ad8a705241d7924","type":"debug","z":"cde349c1477e8ac6","name":"good_messages","active":true,"tosidebar":true,"console":false,"tostatus":false,"complete":"payload","targetType":"msg","statusVal":"","statusType":"auto","x":900,"y":220,"wires":[]},{"id":"3cbcdea3dcd22514","type":"debug","z":"cde349c1477e8ac6","name":"errors and warns","active":true,"tosidebar":true,"console":false,"tostatus":false,"complete":"payload","targetType":"msg","statusVal":"","statusType":"auto","x":910,"y":260,"wires":[]},{"id":"27013483b3180c80","type":"text-classify-text-to-embed","z":"cde349c1477e8ac6","name":"text-classify-text-to-embed","modelPathOrName":"cointegrated/LaBSE-en-ru","orient":"records","x":500,"y":400,"wires":[["02b87ace53cbc9f2","4b51e6c71e39bbd9"],["cc90cd1fecf53b29"]]},{"id":"91ddce031bf7daad","type":"text-classify-light-predictor","z":"cde349c1477e8ac6","name":"text-classify-light-predictor","modelPath":"/tmp","modelName":"lightmodel","orient":"records","x":1050,"y":500,"wires":[["68ae086bd67907a8"],["b3d6ac18d9a39ee1"]]},{"id":"66f8b9407b7cb662","type":"comment","z":"cde349c1477e8ac6","name":"Train with text embedding and classify like 2 in 1","info":"","x":580,"y":60,"wires":[]},{"id":"0bebe1ed963ed6df","type":"comment","z":"cde349c1477e8ac6","name":"Split text embedding and classification","info":"","x":550,"y":320,"wires":[]},{"id":"4b51e6c71e39bbd9","type":"switch","z":"cde349c1477e8ac6","name":"msg.task?","property":"task","propertyType":"msg","rules":[{"t":"eq","v":"train","vt":"str"},{"t":"eq","v":"test","vt":"str"}],"checkall":"true","repair":false,"outputs":2,"x":850,"y":420,"wires":[["d999196e7c73b4ea"],["91ddce031bf7daad"]]},{"id":"02b87ace53cbc9f2","type":"debug","z":"cde349c1477e8ac6","name":"good_messages","active":true,"tosidebar":true,"console":false,"tostatus":false,"complete":"payload","targetType":"msg","statusVal":"","statusType":"auto","x":760,"y":360,"wires":[]},{"id":"cc90cd1fecf53b29","type":"debug","z":"cde349c1477e8ac6","name":"errors and warns","active":true,"tosidebar":true,"console":false,"tostatus":false,"complete":"payload","targetType":"msg","statusVal":"","statusType":"auto","x":750,"y":480,"wires":[]},{"id":"faf312069f01360a","type":"debug","z":"cde349c1477e8ac6","name":"good_messages","active":true,"tosidebar":true,"console":false,"tostatus":false,"complete":"payload","targetType":"msg","statusVal":"","statusType":"auto","x":1300,"y":320,"wires":[]},{"id":"e9c9c62d7b177548","type":"debug","z":"cde349c1477e8ac6","name":"errors and warns","active":true,"tosidebar":true,"console":false,"tostatus":false,"complete":"payload","targetType":"msg","statusVal":"","statusType":"auto","x":1310,"y":380,"wires":[]},{"id":"68ae086bd67907a8","type":"debug","z":"cde349c1477e8ac6","name":"good_messages","active":true,"tosidebar":true,"console":false,"tostatus":false,"complete":"payload","targetType":"msg","statusVal":"","statusType":"auto","x":1280,"y":460,"wires":[]},{"id":"b3d6ac18d9a39ee1","type":"debug","z":"cde349c1477e8ac6","name":"errors and warns","active":true,"tosidebar":true,"console":false,"tostatus":false,"complete":"payload","targetType":"msg","statusVal":"","statusType":"auto","x":1290,"y":540,"wires":[]},{"id":"421ee8b4dcf6320a","type":"inject","z":"cde349c1477e8ac6","name":"Train data sample generator","props":[{"p":"payload"},{"p":"task","v":"train","vt":"str"}],"repeat":"","crontab":"","once":false,"onceDelay":0.1,"topic":"","payload":"[ {\"text\":\"bla-bla\",\"label\":\"talk\"}, {\"text\":\"some message\",\"label\":\"talk\"}, {\"text\":\"I will kill you\",\"label\":\"warning\"}, {\"text\":\"fire at me\",\"label\":\"warning\"}, {\"text\":\"mine field\",\"label\":\"warning\"} ]","payloadType":"json","x":200,"y":380,"wires":[["27013483b3180c80"]]},{"id":"09eeadbcea6f620e","type":"inject","z":"cde349c1477e8ac6","name":"Test data sample generator","props":[{"p":"payload"},{"p":"task","v":"test","vt":"str"}],"repeat":"","crontab":"","once":false,"onceDelay":0.1,"topic":"","payload":"[ {\"text\":\"bla\"}, {\"text\":\"message\"}, {\"text\":\"kill\"}, {\"text\":\"fire\"}, {\"text\":\"mine\"} ]","payloadType":"json","x":190,"y":460,"wires":[["27013483b3180c80"]]},{"id":"d999196e7c73b4ea","type":"text-classify-light-trainer","z":"cde349c1477e8ac6","name":"text-classify-light-trainer","savePath":"/tmp","saveName":"lightmodel","epochCount":"6","orient":"values","x":1070,"y":360,"wires":[["faf312069f01360a"],["e9c9c62d7b177548"]]}]
```
## Thanks
Thanks to  Gabriele Maurina for awesome nodes - [node-red-contrib-machine-learning](https://github.com/GabrieleMaurina/node-red-contrib-machine-learning "node-red-contrib-machine-learning") 