import sys
sys.path.insert(1, '/content/transformer-drg-style-transfer')

import csv
import logging
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from bertviz.bertviz import attention, visualization
from bertviz.bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)

# UNCASED MODEL
logger = logging.getLogger(__name__)

bert_classifier_model_dir =  '/content/drive/MyDrive/misinformation-NLP/models/BERT-SCIFACT-UNCASED' #"/content/BERT-SCIFACT-UNCASED" ## Path of BERT classifier model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

## Model to get the attention weights of all the heads
model = BertModel.from_pretrained(bert_classifier_model_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model.to(device)
model.eval()

max_seq_len=70 # Maximum sequence length 
sm = torch.nn.Softmax(dim=-1) ## Softmax over the batch

def run_multiple_examples(input_sentences, bs=32):
	"""
	This fucntion returns classification predictions for batch of sentences.
	input_sentences: list of strings
	bs : batch_size : int
	"""
	
	## Prepare data for classification
	ids = []
	segment_ids = []
	input_masks = []
	pred_lt = []
	for sen in input_sentences:
		text_tokens = tokenizer.tokenize(sen)
		tokens = ["[CLS]"] + text_tokens + ["[SEP]"]
		temp_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(temp_ids)
		segment_id = [0] * len(temp_ids)
		padding = [0] * (max_seq_len - len(temp_ids))

		temp_ids += padding
		input_mask += padding
		segment_id += padding
		
		ids.append(temp_ids)
		input_masks.append(input_mask)
		segment_ids.append(segment_id)
	
	## Convert input lists to Torch Tensors
	ids = torch.tensor(ids)
	segment_ids = torch.tensor(segment_ids)
	input_masks = torch.tensor(input_masks)
	
	steps = len(ids) // bs
	
	for i in range(steps+1):
		if i == steps:
			temp_ids = ids[i * bs : len(ids)]
			temp_segment_ids = segment_ids[i * bs: len(ids)]
			temp_input_masks = input_masks[i * bs: len(ids)]
		else:
			temp_ids = ids[i * bs : i * bs + bs]
			temp_segment_ids = segment_ids[i * bs: i * bs + bs]
			temp_input_masks = input_masks[i * bs: i * bs + bs]
		
		temp_ids = temp_ids.to(device)
		temp_segment_ids = temp_segment_ids.to(device)
		temp_input_masks = temp_input_masks.to(device)
		
		with torch.no_grad():
			preds = sm(model_cls(temp_ids, temp_segment_ids, temp_input_masks))
		pred_lt.extend(preds.tolist())
	
	return pred_lt

def read_file(path,size):
	with open(path) as fp:
		data = fp.read().splitlines()[:size]
	return data

def get_attention_for_batch(input_sentences, bs=32):
	"""
	This function calculates attention weights of all the heads and
	returns it along with the encoded sentence for further processing.
	
	input sentence: list of strings
	bs : batch_size
	"""
	
	## Preprocessing for BERT 
	ids = []
	segment_ids = []
	input_masks = []
	pred_lt = []
	ids_for_decoding = []
	for sen in input_sentences:
		text_tokens = tokenizer.tokenize(sen)
		tokens = ["[CLS]"] + text_tokens + ["[SEP]"]
		temp_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(temp_ids)
		segment_id = [0] * len(temp_ids)
		padding = [0] * (max_seq_len - len(temp_ids))
		
		ids_for_decoding.append(tokenizer.convert_tokens_to_ids(tokens))
		temp_ids += padding
		input_mask += padding
		segment_id += padding
		
		ids.append(temp_ids)
		input_masks.append(input_mask)
		segment_ids.append(segment_id)
	## Convert the list of int ids to Torch Tensors
	ids = torch.tensor(ids)
	segment_ids = torch.tensor(segment_ids)
	input_masks = torch.tensor(input_masks)
	
	steps = len(ids) // bs
	
	for i in trange(steps+1):
		if i == steps:
			temp_ids = ids[i * bs : len(ids)]
			temp_segment_ids = segment_ids[i * bs: len(ids)]
			temp_input_masks = input_masks[i * bs: len(ids)]
		else:
			temp_ids = ids[i * bs : i * bs + bs]
			temp_segment_ids = segment_ids[i * bs: i * bs + bs]
			temp_input_masks = input_masks[i * bs: i * bs + bs]
		
		temp_ids = temp_ids.to(device)
		temp_segment_ids = temp_segment_ids.to(device)
		temp_input_masks = temp_input_masks.to(device)
		
		with torch.no_grad():
			_, _, attn = model(temp_ids, temp_segment_ids, temp_input_masks)
		
		# Add all the Attention Weights to CPU memory
		# Attention weights for each layer is stored in a dict 'attn_prob'
		for k in range(12):
			attn[k]['attn_probs'] = attn[k]['attn_probs'].to('cpu')
		
		'''
		attention weights are stored in this way:
		att_lt[layer]['attn_probs']['input_sentence']['head']['length_of_sentence']
		'''
		# Concate Attention weights for all the examples in the list att_lt[layer_no]['attn_probs']
		
		if i == 0:
			att_lt = attn
			heads = len(att_lt)
		else:
			for j in range(heads):
				att_lt[j]['attn_probs'] = torch.cat((att_lt[j]['attn_probs'],attn[j]['attn_probs']),0)
		
	
	return att_lt, ids_for_decoding

def process_sentences(input_sentences, att, decoding_ids, threshold=0.25):
	"""
	This function processes each input sentence by removing the top tokens defined threshold value.
	Each sentence is processed for each head.
	
	input_ids: list of strings
	decoding_ids: indexed input_sentnces thus len(input_sentences) == len(decoding_ids)
	threshold: Percentage of the top indexes to be removed
	"""
	# List of None of num_of_layers * num_of_heads to save the results of each head for input_sentences
	
	lt = [None for x in range(len(att) * len(att[0]['attn_probs'][0]))]
	#print(len(lt))
	
	inx = 0
	for i in trange(len(att)): #  For all the layers
		for j in range(len(att[i]['attn_probs'][0])): # For all the heads in the ith Layer
			processed_sen = [None for q in decoding_ids] # List of len(decoding_ids)
			for k in range(len(input_sentences)): # For all the senteces 
				_, topi = att[i]['attn_probs'][k][j][0].topk(len(decoding_ids[k])) # Get top attended ids
				topi = topi.tolist()
				topi = topi[:int(len(topi) * threshold)] 
				## Decode the sentece after removing the topk indexes
				final_indexes = []
				count = 0
				count1 = 0
				tokens = ["[CLS]"] + tokenizer.tokenize(input_sentences[k]) + ["[SEP]"]
				while count < len(decoding_ids[k]):
					if count in topi: # Remove index if present in topk
						while (count + count1 + 1) < len(decoding_ids[k]):
							if "##" in tokens[count + count1 + 1]:
								count1 += 1
							else:
								break
						count += count1
						count1 = 0
					else: # Else add to the decoded sentence
						final_indexes.append(decoding_ids[k][count])
					count += 1
				tmp = tokenizer.convert_ids_to_tokens(final_indexes) # Convert ids to token
				# Convert toknes to sentence
				processed_sen[k] = " ".join(tmp).replace(" ##", "").replace("[CLS]","").replace("[SEP]","").strip()
			lt[inx] = processed_sen # Store sentences for inxth head
			inx += 1
	
	return lt

def get_block_head(processed_sentence_list, lmbd = 0.1):
	"""
	This function calculate classification scores for sentences generated by each head
	and sort them from best to worst.
	score = min(pred) + lmbd / max(pred) + lmbd, lmbd is smoothing param
	pred is list of probability score for each class, for best case pred = [0.5, 0.5] ==> score = 1
	
	it returns sorted list of (Layer, Head, Score)
	"""
	scores = {}
	#scores_1 = {}
	for i in trange(len(processed_sentence_list)): # sentences by each head
		pred = np.array(run_multiple_examples(processed_sentence_list[i]))
		scores[i] = np.mean([(min(x[0], x[1])+lmbd)/(max(x[0], x[1])+lmbd) for x in pred])
		#scores_1[i] = np.mean([abs(max(x[0],x[1]) - min(x[0],x[1])) for x in pred])
	temp = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
	#temp1 = sorted(scores_1.items(), key=lambda kv: kv[1], reverse=False)
	score_lt = [(x // 12, x - (12 * (x // 12)),y) for x,y in temp]
	#score1_lt = [(x // 12, x - (12 * (x // 12)),y) for x,y in temp1]
	return score_lt  #score1_lt

pos_examples_file = arg1 #"/content/scifact_true_dev.txt" #/home/ubuntu/bhargav/data/yelp/sentiment_dev_1.txt"
neg_examples_file = arg2 #"/content/scifact_false_dev.txt" #"/home/ubuntu/bhargav/data/yelp/sentiment_dev_0.txt"
pos_data = read_file(pos_examples_file,100)
neg_data = read_file(neg_examples_file,100)
data = pos_data + neg_data

print(len(pos_data), len(neg_data), len(data))

att, decoding_ids = get_attention_for_batch(data)
sen_list = process_sentences(data, att, decoding_ids)
scores = get_block_head(sen_list)

layer_ = scores[0][0]
head_ = scores[0][1]
print()
print('best_layer: ', layer_, 'best_head: ', head_)
