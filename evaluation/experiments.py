
import pandas as pd
import numpy as np

from tqdm import tqdm
import ast
import os
import json

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



from allennlp.predictors.predictor import Predictor
import allennlp_models.pair_classification


predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/mnli_roberta-2020.06.09.tar.gz", "textual_entailment")
predictor._model = predictor._model.cuda()

test5 = pd.read_csv('./eval_data/evidence_recall/test1_5.tsv', sep='\t')
test5.rename(columns={'entailment': 'label'}, inplace=True)

test1 = pd.read_csv('./eval_data/evidence_recall/test1_1.tsv', sep='\t')
test1.rename(columns={'entailment': 'label'}, inplace=True)

from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('/content/fairseq/roberta.large', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

def get_contra_score(claim, evidence):
	MAX_TOKENS = 512
	sentence_tok = roberta.encode(evidence)
	if len(sentence_tok) > MAX_TOKENS:
			evidence = roberta.decode(sentence_tok[:510])
			print(evidence)
			#continue
	return predictor.predict( hypothesis=claim, premise=evidence)['probs'][:2]

def get_PR(og_df):
	df = og_df.copy()
	yhat = []
	ytrue = []
	ytrue_df = list(df['label'])
	for yt in ytrue_df:
		if yt == 'entailment':
			ytrue.append(0)
		elif yt == 'not_entailment':
			ytrue.append(1)
	
	tqdm.pandas()
	df['pred'] = df.progress_apply( lambda x: np.argmax(get_contra_score( x['sentence2'], x['sentence1']) ), axis=1 )
	yhat = np.array(df['pred'])

	print(precision_score(ytrue, yhat), recall_score(ytrue, yhat), f1_score(ytrue, yhat), accuracy_score(ytrue, yhat))
	return precision_score(ytrue, yhat), recall_score(ytrue, yhat), f1_score(ytrue, yhat), accuracy_score(ytrue, yhat), ytrue, yhat


get_PR(test5)
get_PR(test1)


### COVIDFEVER SCORE

def convert_to_label(x):
	if x == 1:
		return "not_entailment"
	elif x == 0:
		return "entailment"

covidfever = [[convert_to_label(x),convert_to_label(y)] for x,y in zip(yhat5, ytrue5)]
covidfever_df = pd.DataFrame(covidfever, columns=['predicted', 'gold'])

covidfever_df.to_csv('./eval_data/covidfever.tsv', index=True, sep='\t', index_label='index')

count = 0
c = 0
for x,y in zip(open('./eval_data/evidence_recall/test1_5_recall.tsv'),
							 open('./eval_data/covidfever.tsv')):
	
	x = x.strip().split('\t')[-1]
	y = y.strip().split('\t')
	#print(x, y)
	#print(y[1], y[2])
	if y[1]==y[2] and x=='1':
		count = count+1
	c = c+1
print(float(count)/float(c))