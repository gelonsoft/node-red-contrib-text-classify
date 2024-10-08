import sys

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
from sklw import NumpyEncoder
from preprocess_text import preprocess_text

def pretty_output(some):
	result=[]
	for a in some:
		row={}
		for r in a:
			row[r['label']]=r['score']*100
		row=dict(sorted(row.items(), key=lambda item: item[1],reverse=True)[:5])
		result.append(row)
	return result

#read configurations
config = json.loads(unquote(input()))
automl=None
def load():
	try:
		from sklw import SKLW
		return SKLW(path=config['path'],tokenizer_path=config['tokenizerPathOrName'],initial_model_path=config['modelPathOrName'])
	except Exception as e:
		raise e
		return None

while True:
	data=unquote(input())
	if "orient" in data:
		new_config = json.loads(data)
		if new_config.get('modelPath') and new_config.get('modelName'):
			config['path']=os.path.join(new_config['modelPath'], new_config['modelName'])
		if new_config.get('orient'):
			config['orient']=new_config['orient']
		if new_config.get('tokenizerPathOrName'):
			config['tokenizerPathOrName']=new_config['tokenizerPathOrName']
		if new_config.get('modelPathOrName'):
			config['modelPathOrName']=new_config['modelPathOrName']
		automl = load()
		sys.stdout = old_stdout
		print(json.dumps({"state":"parameters applied","config":config}))
		sys.stdout=silent_stdout
		continue

	#read request
	df = pandas.read_json(io.StringIO(data.encode(errors='ignore').decode(encoding='utf-8',errors='ignore')), orient=config['orient'])
	df=df['text'].apply(preprocess_text)
	if automl is None:
		automl = load()
	if automl is None:
		raise('Cannot find model.')
	automl.update()
	try:
		original_result=automl.predict(df.to_list())
	except Exception as e:
		raise
	original_result=pretty_output(original_result)
	sys.stdout = old_stdout
	print(json.dumps({"predict":original_result}), flush=True)
	sys.stdout=silent_stdout
