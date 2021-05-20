import pandas as pd
import numpy as np
from tqdm import tqdm
import re, os, sys, csv
import logging
import random
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

def run_attn_examples(input_sentences, layer, head, bs=128):
	"""
	Returns Attention weights for selected Layer and Head along with ids and tokens
	of the input_sentence
	"""
	ids = []
	ids_to_decode = [None for k in range(len(input_sentences))]
	tokens_to_decode = [None for k in range(len(input_sentences))]
	segment_ids = []
	input_masks = []
	attention_weights = [None for z in input_sentences]
	## BERT pre-processing
	for j,sen in enumerate(tqdm(input_sentences)):
		
		text_tokens = tokenizer.tokenize(sen)
		if len(text_tokens) >= max_seq_len-2:
			text_tokens = text_tokens[:max_seq_len-4]
		tokens = ["[CLS]"] + text_tokens + ["[SEP]"]
		tokens_to_decode[j] = tokens
		temp_ids = tokenizer.convert_tokens_to_ids(tokens)
		ids_to_decode[j] = temp_ids
		input_mask = [1] * len(temp_ids)
		segment_id = [0] * len(temp_ids)
		padding = [0] * (max_seq_len - len(temp_ids))
		
		temp_ids += padding
		input_mask += padding
		segment_id += padding
		
		ids.append(temp_ids)
		input_masks.append(input_mask)
		segment_ids.append(segment_id)
	
	# Convert Ids to Torch Tensors
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
		# Concate Attention weights
		for j in range(len(attn[layer]['attn_probs'])):
			attention_weights[i * bs + j] = (attn[layer]['attn_probs'][j][head][0]).to('cpu')
	
	return attention_weights, ids_to_decode, tokens_to_decode

def prepare_data(aw, ids_to_decode, tokens_to_decode):

	common_words=['is','are','was','were','has','have','had','a','an','the','this','that','these','those','there','how','i','we',
				'he','she','it','they','them','their','his','him','her','us','our', 'and','in','my','your','you', 'will', 'shall']
	common_words_tokens = tokenizer.convert_tokens_to_ids(common_words)
	not_to_remove_ids = tokenizer.convert_tokens_to_ids(["[CLS]","[SEP]", ".", "?", "!"])
	not_to_remove_ids += common_words_tokens

	#out_sen = [None for i in range(len(aw))]
	all_top_words = []
	for i in trange(len(aw)):
		topv, topi = aw[i].topk(len(ids_to_decode[i]))
		#print(ids_to_decode[i].index(0))
		topv, topi = aw[i].topk(ids_to_decode[i].index(0)) #aw[i].topk(k=2)
		#topv, topi = aw[i].topk(k=2)
		topi = topi.tolist()
		topv = topv.tolist()
		#print(topi)
		#print(topv)
		# print(i,train_0[i])
		# print(tokens_to_decode[i])
		# print("Original Top Indexes = {}".format(topi))
		topi = [topi[j] for j in range(len(topi)) if ids_to_decode[i][topi[j]] not in not_to_remove_ids] # remove noun and common words
		#print("After removing Nouns = {}".format(topi))
		topi = [topi[j] for j in range(len(topi)) if "##" not in tokens_to_decode[i][topi[j]]] # Remove half words
		#print("After removing Half-words = {}".format(topi))
		
		top_words = []
		printed=0 
		for j in range(0, len(topi)):
		
			top_word = tokens_to_decode[i][topi[j]]
			if len(top_word) >= 3 and "virus" not in top_word \
				and "corona" not in top_word and printed<3:
				printed+=1
				#print(top_word)
				top_words.append(top_word)
		all_top_words.append(top_words)

	return all_top_words
	


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)

input_file = arg1
outfile = arg2
layer_ = 6
head_ = 8

max_seq_len=70 # Maximum sequence length 
sm = torch.nn.Softmax(dim=-1) ## Softmax over the batch

# UNCASED MODEL
logger = logging.getLogger(__name__)

bert_classifier_model_dir =  '/content/drive/MyDrive/misinformation-NLP/models/BERT-SCIFACT-UNCASED' ## Path of BERT classifier model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

## Model for performing Classification
model_cls = BertForSequenceClassification.from_pretrained(bert_classifier_model_dir, num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # replace scibert ?
model_cls.to(device)
model_cls.eval()

## Model to get the attention weights of all the heads
model = BertModel.from_pretrained(bert_classifier_model_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model.to(device)
model.eval()

train_0 = [c.replace("\"", "").strip() for c in input_file.readlines()]
aw, ids_to_decode, tokens_to_decode = run_attn_examples(train_0, layer=layer_, head=head_, bs=128)
top_words = prepare_data(aw, ids_to_decode, tokens_to_decode)

outfile = '/content/drive/MyDrive/misinformation-NLP/results/additional_1k.csv'

sources = list(claims['source_url'])
flairs = list(claims['flair'])
with open(outfile, "w") as out_fp:

	writer = csv.writer(out_fp, delimiter=",")
	writer.writerow(['claim', 'keyword', 
					'src_url', 'flair'])
	for i in range(0, len(top_words)):
		for j in range(0, len(top_words[i])):
			claim = train_0[i]
			keyword = top_words[i][j]
			src_url = sources[i]
			flair = flairs[i]
			writer.writerow([claim, keyword, src_url, flair])


