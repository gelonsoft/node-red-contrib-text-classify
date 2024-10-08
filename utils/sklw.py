import os
import json
import numpy as np
import sys
import evaluate
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, \
	AutoModelForSequenceClassification, TextClassificationPipeline

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')

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
	def __init__(self, path, tokenizer_path,initial_model_path=None,id2label=None):
		self.path = path
		self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)

		if id2label is not None:
			self.model= AutoModelForSequenceClassification.from_pretrained(initial_model_path,num_labels=len(id2label),id2label=id2label,label2id={y:x for x,y in id2label.items()})
			self.pipe=TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, top_k=5)
		else:
			try:
				self.model= AutoModelForSequenceClassification.from_pretrained(self.path)
				self.last = os.stat(self.path+"/config.json").st_mtime
				self.pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, top_k=5)
			except Exception as e:
				print(e)
				self.model= AutoModelForSequenceClassification.from_pretrained(initial_model_path)
				self.pipe= TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, top_k=5)

	def fit(self, datasets):
		tokenizer=self.tokenizer
		def preprocess_function(examples):
			return tokenizer(examples["text"], truncation=True)

		datasets["train"]=datasets["train"].map(preprocess_function,batched=True)
		datasets["test"]=datasets["test"].map(preprocess_function,batched=True)
		data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
		accuracy = evaluate.load("accuracy")
		def compute_metrics(eval_pred):
			predictions, labels = eval_pred
			predictions = np.argmax(predictions, axis=1)
			return accuracy.compute(predictions=predictions, references=labels)

		training_args = TrainingArguments(
			output_dir=self.path,
			learning_rate=2e-5,
			per_device_train_batch_size=16,
			per_device_eval_batch_size=16,
			num_train_epochs=2,
			weight_decay=0.01,
			evaluation_strategy="no",
			save_strategy="no",
			load_best_model_at_end=True,
			push_to_hub=False,
		)

		trainer = Trainer(
			model=self.model,
			args=training_args,
			train_dataset=datasets["train"],
			eval_dataset=datasets["test"],
			tokenizer=self.tokenizer,
			data_collator=data_collator,
			compute_metrics=compute_metrics,
		)
		#best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
		#for n, v in best_run.hyperparameters.items():
		#	setattr(trainer.args, n, v)

		trainer.train()

		dir = os.path.dirname(self.path)
		if not os.path.isdir(dir):
			os.makedirs(dir, exist_ok=True)
		trainer.save_model(output_dir=self.path)

	def predict(self, x):
		return self.pipe(x)

	def update(self):
		modified = os.stat(self.path+"/config.json").st_mtime
		if(modified > self.last):
			self.last = modified
			self.model= AutoModelForSequenceClassification.from_pretrained(self.path)
			self.pipe= TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, top_k=5)