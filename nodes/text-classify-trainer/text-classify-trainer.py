import sys

old_stdout = sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

from datasets import Dataset
import base64
import json
import pandas
import io

import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')
from sklw import SKLW
from preprocess_text import preprocess_text

REG_DETECTORS = ['text-classify-trainer']

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
		if new_config.get('tokenizerPathOrName'):
			config['tokenizerPathOrName']=new_config['tokenizerPathOrName']
		if new_config.get('modelPathOrName'):
			config['modelPathOrName']=new_config['modelPathOrName']
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
			continue
	else:
		try:
			#load data from request
			df = pandas.read_json(io.StringIO(json.dumps(data).encode(errors='ignore').decode(encoding='utf-8',errors='ignore')), orient=config['orient'])
		except Exception as e:
			print(e)
			continue

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
	content=json.dumps({"state":"training completed","automl":config['automl']})
	sys.stdout = old_stdout
	print(base64.b64encode(content.encode()).decode('utf-8')+"\t\t\t\n",flush=True)
	sys.stdout=silent_stdout

	print("Done")
