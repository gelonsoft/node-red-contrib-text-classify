import sys

old_stdout=sys.stdout
silent_stdout = sys.stderr
sys.stdout = silent_stdout

import json
import pandas
import io
from urllib.parse import unquote
from official.nlp import optimization #Do not del
import tensorflow_text as text  #Do not del
import tensorflow as tf
tf.get_logger().setLevel('ERROR')



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')
from sklw import NumpyEncoder
from preprocess_text import preprocess_text
from my_model import GET_LEN_SEQ

def get_pretty_output(labels,result):
	a=dict(zip(labels,result))
	return {k: str(v*100) for k, v in sorted(a.items(), key=lambda item: item[1],reverse=True)[:5]}

#read configurations
config = json.loads(unquote(input()))
model=None
def load():
	try:
		from sklw import SKLW
		return SKLW(path=config['path'])
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
		model = load()
		sys.stdout = old_stdout
		print(json.dumps({"state":"parameters applied","config":config}))
		sys.stdout=silent_stdout
		continue

	#read request
	features = pandas.read_json(io.StringIO(data.encode(errors='ignore').decode(encoding='utf-8',errors='ignore')), orient=config['orient'])
	x=features['x'].apply(preprocess_text)
	if model is None:
		model = load()
	if model is None:
		raise('Cannot find model.')
	model.update()
	if os.getenv('TC_USE_HUBBLE_HUG','0')=='1':
		from datasets import Dataset
		original_result=model.predict(dict(model.tokenizer(Dataset.from_pandas(pandas.DataFrame(x))["x"], padding='max_length', truncation=True, max_length=GET_LEN_SEQ(), return_tensors='tf')))
	else:
		original_result=model.predict(x)
	sys.stdout = old_stdout
	print(json.dumps({"predict":[get_pretty_output(model.labels,z) for z in original_result]},cls=NumpyEncoder), flush=True)
	sys.stdout=silent_stdout
