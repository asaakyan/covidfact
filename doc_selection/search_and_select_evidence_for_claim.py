import pandas as pd
import os
from google.colab import drive
import requests

from sentence_transformers import SentenceTransformer, util

from sentence_transformers.util import semantic_search
#Code for Google Search adapted from https://github.com/tuhinjubcse/DeSePtion-ACL2020/blob/master/src/doc/getDocuments.py
import sys
import os
import json
import ast
from time import sleep
from googleapiclient.discovery import build #google-api-python-client
import pprint
import nltk
import ast
import unicodedata
import time
import logging
import tldextract
from tqdm import tqdm
import time

import numpy as np
import requests
import re
import sys
import pandas as pd
from os import path
from bs4 import BeautifulSoup

import traceback

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import html2text


logger = logging.getLogger(__name__)

class GoogleConfig:
		def __init__(self, api_key=None, cse_id=None, site='', num=5, max_docs=2):
				if 'API_KEY' in os.environ:
						api_key = os.environ['API_KEY']
				if 'SEARCH_ID' in os.environ:
						cse_id = os.environ['SEARCH_ID']
				assert(api_key is not None and cse_id is not None)
				
				self.api_key = api_key
				self.cse_id = cse_id
				self.site = site
				self.num = num
				self.max_docs = max_docs

		def __str__(self):
				return 'GoogleConfig(api_key={}, cse_id={}, site={}, num={}, max_docs={}'.format(self.api_key,
																																												 self.cse_id,
																																												 self.site,
																																												 self.num,
																																												 self.max_docs)
				
def google_search(search_term, api_key, cse_id, **kwargs):
		service = build("customsearch", "v1", developerKey=api_key)
		res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
		if 'items' in res:
				return res['items']
		else:
				return []
				
def getDocumentsForClaimFromGoogle(claim, google_config):    
		results = google_search(google_config.site+' '+claim, google_config.api_key,
														google_config.cse_id, num=google_config.num)
		logger.info(google_config)
		res = []
		for result in results:
			#return result
			try:
				res.append(result['link'])
			except:
				res.append('notfound')
			#print(result['formattedUrl'])
		return res

h = html2text.HTML2Text()
h.ignore_links = True
h.no_wrap_links = True
h.ignore_images = True

def scrape_link(url, sent, h=h):
		print("*"*20)
		print("PARSING: ", url)
		#html = requests.get(url, timeout=10).text
		foundtitle=False
			
		try:

			html = requests.get(url, timeout=10).text
			soup = BeautifulSoup(html, 'html.parser')
			try:
				title = soup.find('title').string
				if sent.lower().replace(" ", "") in title.lower().replace(" ", ""):
					foundtitle = True
			except:
				foundtitle = False

			text = h.handle(html)
			
			text = re.sub(r'\n+', '\n', text)
			text = re.sub(r'#+', '#', text)
			text = re.sub(r'\*+', '', text)
			text = re.sub(r' \w*@\w*', '', text) # remove everything containing @
			text = re.sub(r' \[\w*\]', '', text) # remove things inside []

			#text = text.replace("\n\n", "\n")
			text = text.replace(".\n", ". ")
			#text = text.replace("\n", ". ")
			text = text.strip()
			text = text.replace("_", "")

			text = " ".join(text.split())
			tokenized_text = sent_tokenize(text)
			#sentences = tokenized_text
			sentences = set()


			for tok in tokenized_text:
				toks = tok.split("#") # split by # in case claim has it
				if len(toks) > 1:
					tok = toks[np.argmax([len(toks) for t in toks])]
				toks = tok.split("|") # split by | in case claim has it
				if len(toks) > 1:
					tok = toks[np.argmax([len(toks) for t in toks])]
				if ''.join(filter(str.isalpha, sent.lower())) in ''.join(filter(str.isalpha, tok.lower())):
					continue
				sentences.add(tok)
			
			if len(sentences) == 0:
				sentences = ['COULDNOTGETSENTS']
				foundtitle=False

		except Exception as inst:
			print(type(inst))    # the exception instance
			print(inst.args)     # arguments stored in .args
			print(inst)    
			sentences = ['COULDNOTGETSENTS']
			foundtitle=False

		return list(sentences), foundtitle

def select_evidence(model, claim, corpus, top_k=5):
	# returns topk sentences by cosine similarity

	embeddings1 = model.encode(claim, convert_to_tensor=True)
	embeddings2 = model.encode(corpus, convert_to_tensor=True)

	result = semantic_search(embeddings1, embeddings2, top_k=top_k)  
	rel_sents = [ (corpus[corpid['corpus_id']], corpid['score']) for corpid in result[0] ] 

	return rel_sents


def get_evidence_for_claim(claim, search_res, model):
	
	corpus = set()

	for sr in search_res:
		if not ".pdf" in sr: # do not process pdf
			new_sents, found_title = scrape_link(sr, claim)
			if found_title: # if we find exact source of the claim, no need to process anything more
				evidence = select_evidence(model, claim, new_sents)
				# print(claim)
				# print(evidence[0])
				# print("+++"*100)
				return evidence
			else:
				for s in new_sents:
					corpus.add(s)

	evidence = select_evidence(model, claim, list(corpus))

	return evidence

if __name__ == "__main__":
	api_key = 0000
	cse_id = 0000

	google_config = GoogleConfig(api_key, cse_id, num=5)
	search_results = getDocumentsForClaimFromGoogle(claim, google_config)
	model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')
	
	print(get_evidence_for_claim(claim, search_results, model))

