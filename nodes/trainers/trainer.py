import sys
from datasets import Dataset

old_stdout=sys.__stdout__
silent_stdout = sys.stderr
sys.stdout = silent_stdout

import json
import pandas
import io
from urllib.parse import unquote

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')
from sklw import SKLW
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
		if new_config.get('tokenizerPathOrName'):
			config['tokenizerPathOrName']=new_config['tokenizerPathOrName']
		if new_config.get('modelPathOrName'):
			config['modelPathOrName']=new_config['modelPathOrName']
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
		pass

	automl = None

	if config['automl'] == 'text-classify-trainer':
		df['text']=df['text'].apply(preprocess_text)
		categor=pandas.Categorical(df['label'])
		id2label=dict( enumerate(categor.categories ) )
		df['label']=pandas.DataFrame(categor.codes)
		categor=None
		datasets=Dataset.from_pandas(df).train_test_split(test_size=0.2)
		df=None
		automl = SKLW(path=save,tokenizer_path=config['tokenizerPathOrName'],initial_model_path=config['modelPathOrName'],id2label=id2label)
		try:
			automl.fit(datasets)
		except Exception as e:
			raise()

	sys.stdout = old_stdout
	print(json.dumps({"state":"training completed","automl":config['automl']}), flush=True)
	sys.stdout=silent_stdout

	print("Done")
