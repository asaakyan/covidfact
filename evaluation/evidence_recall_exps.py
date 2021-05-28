import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import os
import json
from scipy import stats

def get_n_evid(evidence_batch1, evidence_batch2, n=5):
	all_evid = {}
	for idx, x in evidence_batch1.iterrows():

		claim = ''.join( char for char in x['claim'].capitalize() if len(char.encode('utf-8')) < 3) 
		evid_list = ast.literal_eval(x['evidence'])[:n]
		evids = [e[0] for e in evid_list]
		evids = [''.join(char for char in e if len(char.encode('utf-8')) < 3) for e in evids]
		all_evid[claim] = evids
	for idx, x in evidence_batch2.iterrows():
		claim = ''.join( char for char in x['claim'].capitalize() if len(char.encode('utf-8')) < 3) 
		# if 'Restriction of' in x['claim']:
		#   print(x['claim'])
		evid_list = ast.literal_eval(x['evidence'])[:n]
		evids = [e[0] for e in evid_list]
		evids = [''.join(char for char in e if len(char.encode('utf-8')) < 3) for e in evids]
		all_evid[claim] = evids
	return all_evid

def get_retrieved_evid(test_set_fname, positive_retrieved_evid, refuted_retrieved_evid, n):
	retrieved_evid = {}
	test_data_df = pd.read_csv(test_set_fname, sep='\t')
	for claim in list(test_data_df['sentence2']):
		claim = claim.capitalize()
		if claim in positive_retrieved_evid:
			retrieved_evid[claim] = positive_retrieved_evid[claim][:n]
		elif claim in refuted_retrieved_evid:
			retrieved_evid[claim] = refuted_retrieved_evid[claim][:n]
		else:
			print(claim)
			print("NOT FOUNDD")
			break
	assert len(retrieved_evid) == len(test_data_df)
	return retrieved_evid

def get_test_gold(dataset_fname, test_set_fname):

	test_gold = {}

	with open(dataset_fname) as f:
		gold_data = json.load(f)
	#print(len(gold_data))
	test_data_df = pd.read_csv(test_set_fname, sep='\t')
	for claim in list(test_data_df['sentence2']):
		test_gold[claim] = gold_data[claim]['evidence']
	
	return test_gold 

fname = './eval_data/evidence_recall/test1.tsv'

evidence_batch1 = pd.read_csv('./eval_data/evidence_recall/siteMAtchEVIDENCE_LINK_v2.csv')
evidence_batch2 = pd.read_csv('./eval_data/evidence_recall/siteMAtchEVIDENCE_LINKS_ADDITIONAL_1k.csv')

positive_retrieved_evid = get_n_evid(evidence_batch1, evidence_batch2)

with open("./eval_data/evidence_recall/refuted_retrieved_evidence.json") as f:
	refuted_retrieved_evid = json.load(f)

evid5 = get_retrieved_evid(fname, positive_retrieved_evid, refuted_retrieved_evid, 5)
evid3 = get_retrieved_evid(fname, positive_retrieved_evid, refuted_retrieved_evid, 3)
evid1 = get_retrieved_evid(fname, positive_retrieved_evid, refuted_retrieved_evid, 1)

gold_fname = './eval_data/evidence_recall/DATASET_4k.json' #/content/drive/MyDrive/misinformation-NLP/tmp/DATASET/DATASET_4k.json'
test_set_fname = './eval_data/evidence_recall/test1.tsv'  #/content/drive/MyDrive/misinformation-NLP/tmp/DATASET/rte-covid-tuhin/test1.tsv'
test_gold = get_test_gold(gold_fname, test_set_fname)

def get_PR(all_evid, human_evid):
	# based on https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

	recall_overall = 0
	precision = 0
	recall = 0
	precision_by_len = 0

	r_hits = 0
	p_hits = 0
	r_overall_hits = 0

	for hevid in human_evid:

		
		gold = human_evid[hevid]#['evidence']
		predicted = all_evid[hevid.capitalize()]



		predicted = [p.strip() for p in predicted]

		r_hits += 1
		at_least_one = False 
		for g in gold:
			
			if g.strip() in predicted or g in predicted:
				at_least_one = True

		if at_least_one:
			recall += 1
			all_evid[hevid.capitalize()] = {'evid': all_evid[hevid.capitalize()], 'recall':1 }
		else:
			all_evid[hevid.capitalize()] = {'evid': all_evid[hevid.capitalize()], 'recall':0 }
	
		for pred in predicted:
			p_hits += 1
			if pred in gold:
				precision += 1
		
	pr = precision/p_hits
	rec = recall/r_hits
	print(pr)
	print(rec)
	f1 = 2.0 * pr * rec / (pr + rec)
	print(f1)

get_PR(evid5, test_gold)
get_PR(evid3, test_gold)
get_PR(evid1, test_gold)

# save tsv for varacity + retrieval

# def save_tsv(og_test_fname, output_fname, evid_json):

# 	dev_df_data = []

# 	og_test_df = pd.read_csv(og_test_fname, sep='\t')
# 	for idx, row in og_test_df.iterrows():
# 		claim = row['sentence2'].capitalize()
# 		label = row['label']

# 		evidence = evid_json[claim]['evid']
# 		recall = evid_json[claim]['recall']

# 		dev_df_data.append([" ".join(evidence).replace('\n', ' ').replace('"', '') , claim.capitalize(), label, recall])
# 	dev_df = pd.DataFrame(dev_df_data, columns=['sentence1', 'sentence2', 'label', 'recall'])

# 	display(dev_df)
# 	dev_df.to_csv(output_file_path, index=True, sep='\t', index_label='index')
# 	print('saved successfully')

# og_test_fname = '/content/drive/MyDrive/misinformation-NLP/tmp/DATASET/rte-covid-tuhin/test1.tsv'

# output_file_path = '/content/drive/MyDrive/misinformation-NLP/tmp/DATASET/rte-covid-tuhin/evidence_recall/test1_5_recall.tsv'

# save_tsv(og_test_fname, output_file_path, evid5)

# now save to tsv for the rTE task

# def save_tsv(og_test_fname, output_fname, evid_json):

# 	dev_df_data = []

# 	og_test_df = pd.read_csv(og_test_fname, sep='\t')
# 	for idx, row in og_test_df.iterrows():
# 		claim = row['sentence2'].capitalize()
# 		label = row['label']
# 		evidence = evid_json[claim]
# 		# if len(evidence) != 5:
# 		#   evidence = ['not found']
# 		#   # print(len(evidence))
# 		#   # print(evidence)
# 		#assert len(evidence) == n
# 		dev_df_data.append([" ".join(evidence).replace('\n', ' ').replace('"', '') , claim.capitalize(), label])
# 	dev_df = pd.DataFrame(dev_df_data, columns=['sentence1', 'sentence2', 'label'])

# 	#display(dev_df)
# 	dev_df.to_csv(output_file_path, index=True, sep='\t', index_label='index')
# 	print('saved successfully')

