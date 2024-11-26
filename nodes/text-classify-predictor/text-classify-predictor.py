import sys

old_stdout = sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import base64
import json
import pandas
import io

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')
from preprocess_text import preprocess_text

if os.environ.get('DISABLE_SSL_VERIFY', "0") == "1":
	import ssl
	ssl._create_default_https_context = ssl._create_unverified_context

def pretty_output(some):
	result=[]
	for a in some:
		row={}
		for r in a:
			row[r['label']]=r['score']*100
		row=dict(sorted(row.items(), key=lambda item: item[1],reverse=True)[:5])
		result.append(row)
	return result


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

automl=None
def load():
	try:
		from sklw import SKLW
		return SKLW(path=config['path'],tokenizer_path=config['tokenizerPathOrName'],initial_model_path=config['modelPathOrName'])
	except Exception as e:
		raise e
		return None

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
		if new_config.get('modelPath') and new_config.get('modelName'):
			config['path']=os.path.join(new_config['modelPath'], new_config['modelName'])
		if new_config.get('orient'):
			config['orient']=new_config['orient']
		if new_config.get('tokenizerPathOrName'):
			config['tokenizerPathOrName']=new_config['tokenizerPathOrName']
		if new_config.get('modelPathOrName'):
			config['modelPathOrName']=new_config['modelPathOrName']
		automl = load()
		content=json.dumps({"state":"parameters applied","config":config})
		sys.stdout = old_stdout
		print(base64.b64encode(content.encode()).decode('utf-8')+"\t\t\t\n",flush=True)
		sys.stdout=silent_stdout
		continue

	#read request
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
	try:
		df=df['text'].apply(preprocess_text)
		if automl is None:
			automl = load()
		if automl is None:
			raise('Cannot find model.')
		automl.update()
		content=json.dumps({"predict":pretty_output(automl.predict(df.to_list()))})
		sys.stdout = old_stdout
		print(base64.b64encode(content.encode()).decode('utf-8')+"\t\t\t\n",flush=True)
		sys.stdout=silent_stdout
	except Exception as e:
		print(e)
		continue
