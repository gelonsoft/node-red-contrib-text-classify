#SciKit-Learn models' wrapper

import pickle
import os
import json
import numpy as np
import sys
from keras.models import load_model

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')
from my_model import load_tokenizer

class NumpyEncoder(json.JSONEncoder):
	""" Special json encoder for numpy types """
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

class SKLW:
	def __init__(self, path, model=None, labels=None):
		self.path = path
		if os.getenv('TC_USE_HUBBLE_HUG','0')=='1':
			self.tokenizer=load_tokenizer()
		else:
			self.tokenizer=None
		if model is not None:
			self.model = model
			self.labels = labels
		else:
			self.last = os.stat(self.path).st_mtime
			self.model= load_model(self.path, compile=False)
			#self.model = pickle.load(open(self.path, "rb"))
			self.labels = pickle.load(open(self.path+".labels.txt", "rb"))

	def fit(self, x, y=None):
		if y is not None:
			self.model.fit(x=x, y=y,validation_split=0.2)
		else:
			self.model.fit(x=x,validation_split=0.2)

		#print("Saving model to "+self.path)
		dir = os.path.dirname(self.path)
		if not os.path.isdir(dir):
			os.makedirs(dir, exist_ok=True)

		#pickle.dump(self.model, open(self.path, "wb"))
		#print("Saving model to "+self.path)
		self.model.save(self.path)
		pickle.dump(self.labels, open(self.path+".labels.txt", "wb"))

	def predict(self, x):
		return self.model.predict(x) #.tolist()

	def fit_predict(self, x, roles):
		self.model.fit(x=x.loc[:, x.columns != roles['target']],y=x[roles['target']],validation_split=0.2)
		dir = os.path.dirname(self.path)
		if not os.path.isdir(dir):
			os.makedirs(dir, exist_ok=True)

		#pickle.dump(self.model, open(self.path, "wb"))
		self.model.save(self.path)
		pickle.dump(self.labels, open(self.path+".labels.txt", "wb"))

	def update(self):
		modified = os.stat(self.path+".labels.txt").st_mtime
		if(modified > self.last):
			self.last = modified
			#self.model = pickle.load(open(self.path, "rb"))
			self.model= load_model(self.path, compile=False)
			self.labels=pickle.load(open(self.path+".labels.txt", "rb"))