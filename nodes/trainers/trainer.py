import sys

old_stdout=sys.stdout
silent_stdout = sys.stderr
sys.stdout = silent_stdout

import json
import pickle
import pandas
import io

from urllib.parse import unquote
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')
from sklw import SKLW
from my_model import create_new_model
from preprocess_text import preprocess_text

REG_DETECTORS = ['text-classify-trainer']

#read configurations
config = json.loads(unquote(input()))
save = config['save']


while True:
	#read request
	data = unquote(input())
	if "orient" in data:
		new_config = json.loads(data)
		if new_config.get('savePath') and new_config.get('saveName'):
			config['save']=os.path.join(new_config['savePath'], new_config['saveName'])
		if new_config.get('orient'):
			config['orient']=new_config['orient']

		save = config['save']
		sys.stdout = old_stdout
		print(json.dumps({"state":"parameters applied","config":config}), flush=True)
		sys.stdout=silent_stdout
		continue


	try:
		#load data from request
		df = pandas.read_json(io.StringIO(data.encode(errors='ignore').decode(encoding='utf-8',errors='ignore')), orient=config['orient'])

	except Exception as e:
		print(e)
		#lead file specified in the request

		df = pandas.read_csv(json.loads(data)['file'])

	labels=None

	if config['automl'] in REG_DETECTORS:
		x=df['x'].apply(preprocess_text)
		categor=pandas.Categorical(df['y'])
		y=pandas.DataFrame(categor.codes)
		#df.to_csv("d:/a/df.csv")
		#y.to_csv("d:/a/y.csv")
		df=None
		labels=categor.categories.to_list()

	automl = None

	if config['automl'] == 'text-classify-trainer':
		automl = SKLW(path=save, model=create_new_model(len(labels)),labels=labels) #name = 'reg',#metric = config['metric']))

	try:
		#train model
		if os.getenv('TC_USE_HUBBLE_HUG','0')=='1':
			from datasets import Dataset
			automl.fit(dict(automl.tokenizer(Dataset.from_pandas(pandas.DataFrame(x))["x"], padding=True, truncation=True, max_length=64, return_tensors='tf')),y)
		else:
			automl.fit(pandas.DataFrame(x),y)
	except Exception as e:
		raise()

	sys.stdout = old_stdout
	print(json.dumps({"state":"training completed","automl":config['automl']}), flush=True)
	sys.stdout=silent_stdout

	print("Done")
