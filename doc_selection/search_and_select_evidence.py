import pandas as pd
import os
from google.colab import drive

!pip install tldextract
drive.mount('/content/drive')
dir = "/content/drive/My Drive/misinformation-NLP/"
os.listdir(dir)

datapath = '/content/drive/MyDrive/misinformation-NLP/results/highscores.csv'
datapath = '/content/drive/MyDrive/misinformation-NLP/results/best3contra_additional_1k.csv'
df = pd.read_csv(datapath)
df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
df.sort_values(by=['contra_score'], ascending=False, ignore_index=True, inplace=True)
df.drop_duplicates(subset=['claim'], inplace=True)
df

import requests

# Tuhin's code
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


# Let us first filter claims by site rank

##### SITERANK METHOD BASED ON https://github.com/prahladyeri/siterank/blob/master/siterank/siterank.py ####
import urllib.request
import sys, os
import argparse
import xml.etree.ElementTree as ET

def get_siterank(url):
	#https://stackoverflow.com/questions/3676376/fetching-alexa-data
	turl = "http://data.alexa.com/data?cli=10&url=" + url
	req = urllib.request.Request(turl)
	with urllib.request.urlopen(req) as fp:
		output = fp.read().decode('utf-8')
		tree = ET.ElementTree(ET.fromstring(output))
		root = tree.getroot()
		sd = root.find("SD")
		if sd != None:
			rnk = int(sd.find("POPULARITY").get("TEXT"))
			return int(rnk)
			#ranks[url] = rnk
		else:
			print("Not found: " + url)
			return 1000000

from tqdm import tqdm
# tqdm.pandas()
df['siterank'] = df['source'].progress_apply(get_siterank)
#df.to_csv('/content/drive/MyDrive/misinformation-NLP/tmp/dfRANKS_additional_1k.csv')

####  TESTING ALREADY LABELED FAKE  SITES
##### SITERANK METHOD BASED ON https://github.com/prahladyeri/siterank/blob/master/siterank/siterank.py ####
import urllib.request
import sys, os
import argparse
import xml.etree.ElementTree as ET
import numpy as np

def get_siterank(url):
	#https://stackoverflow.com/questions/3676376/fetching-alexa-data
	turl = "http://data.alexa.com/data?cli=10&url=" + url
	req = urllib.request.Request(turl)
	with urllib.request.urlopen(req) as fp:
		output = fp.read().decode('utf-8')
		tree = ET.ElementTree(ET.fromstring(output))
		root = tree.getroot()
		sd = root.find("SD")
		if sd != None:
			rnk = int(sd.find("POPULARITY").get("TEXT"))
			return int(rnk)
			#ranks[url] = rnk
		else:
			#print("Not found: " + url)
			return 1000000

from tqdm import tqdm
import matplotlib.pyplot as plt
# get_siterank('worldinformation24.info')

# ranks = []
# with open('domains.txt', 'r') as domains:
# 	for line in tqdm(domains):
# 		rank = get_siterank(line)
# 		if rank <= 50000:
# 			print(line, rank)
# 		ranks.append(rank)
# print(np.mean(ranks))
# print(min(ranks))

# ranks = []
# with open('domainsred.txt', 'r') as domains:
# 	for line in tqdm(domains):
# 		rank = get_siterank(line)
# 		if rank <= 50000:
# 			print(line, rank)
# 		ranks.append(rank)
# print(np.mean(ranks))
# print(min(ranks))

# import  ast
# with open('domainsred.txt', 'r') as domainsred:
# 	red = set([d.strip() for d in domainsred.readlines()])
	
# with open('domains.txt', 'r') as domainsblack:
# 	black = set([d.strip() for d in domainsblack.readlines()])

# with open('sources.txt', 'r') as sources:
# 	src = ast.literal_eval(sources.read())

# print(red)
# print(black)
# print(src)
# print(src.intersection(black))
# print(src.intersection(red))

import numpy as np
# print(np.mean(ranks))

df = pd.read_csv('/content/drive/MyDrive/misinformation-NLP/tmp/dfRANKS_additional_1k.csv')
highranks = df[df['siterank']<50000]
# highranks

# Now let us obtain a file with link search results for these claims

def get_search_results(df, google_config):
	claim_search_res = dict()
	for ind, row in tqdm(df.iterrows()):
		try:
			search_results = getDocumentsForClaimFromGoogle(row['claim'], google_config)
			claim_search_res[row['claim']] = search_results
			#time.sleep(1)
		except: 
			time.sleep(70)
	return claim_search_res

# api_key = ''
# cse_id = ''
google_config = GoogleConfig(api_key, cse_id, num=5)

search_results5 = get_search_results(highranks, google_config)

import json  

# outfile = "/content/drive/MyDrive/misinformation-NLP/tmp/req_search_results10.json"
# outfile = "/content/drive/MyDrive/misinformation-NLP/tmp/search_results_5.json"
outfile = "/content/drive/MyDrive/misinformation-NLP/tmp/search_results_5_additional_1k.json"

with open(outfile, "w") as outfile:  
		#json.dump(search_results5, outfile)

import json  

#srfile10 = '/content/drive/MyDrive/misinformation-NLP/tmp/search_results.json'
#srfile5 = "/content/drive/MyDrive/misinformation-NLP/tmp/search_results_5.json"
srfile5 = "/content/drive/MyDrive/misinformation-NLP/tmp/search_results_5_additional_1k.json"

with open(srfile5) as f:
	search_results5 = json.load(f)

# with open(srfile10) as f:
#   search_results10 = json.load(f)

df = pd.read_csv('/content/drive/MyDrive/misinformation-NLP/tmp/dfRANKS_additional_1k.csv')
highranks = df[df['siterank']<50000]
# highranks

# Now we want to select only those claims which had their website found

def get_sitename(link):
	extract = tldextract.extract(link)
	sitename = extract.domain +"."+ extract.suffix
	return sitename

def get_overlap_stats(df, search_results):

	SITES = []
	siteMatch = []

	for ind, row in tqdm(df.iterrows()):

		source = row['source']
		claim = row['claim']

		srs = search_results[claim]
		if len(srs)==0:
			SITES.append('none')
			siteMatch.append(False)
			continue

		sitenames = [get_sitename(x) for x in srs]
		SITES.append(sitenames)
		source_name = get_sitename(source)
		SM = source_name in sitenames
		siteMatch.append(SM) # check if site name has a match
	
	return siteMatch, SITES

matches = highranks.copy()
matches['siteMatch'], matches['SITES'] = get_overlap_stats(highranks, search_results5)

matches = matches[matches['siteMatch']==True]
# matches


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

# sent = 'Sequential infection with influenza A virus followed by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) leads to more severe disease and encephalitis in a mouse model of COVID-19'
# link = 'https://www.biorxiv.org/content/10.1101/2020.10.13.334532v1'
# scrape_link(link, sent)

# !pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')

from sentence_transformers.util import semantic_search

def select_evidence(model, claim, corpus, top_k=5):
	# returns topk sentences by cosine similarity

	# print(claim)
	# print(corpus)

	embeddings1 = model.encode(claim, convert_to_tensor=True)
	embeddings2 = model.encode(corpus, convert_to_tensor=True)

	result = semantic_search(embeddings1, embeddings2, top_k=top_k)  
	rel_sents = [ (corpus[corpid['corpus_id']], corpid['score']) for corpid in result[0] ] 

	return rel_sents

#Compute embedding for both lists
# sentence = 'Nasal-Swab Testing Misses Patients with Low SARS-CoV-2 Viral Loads'

# corpus = ['Reinfection could not occur in SARS-CoV-2 infected rhesus macaques',
# 							'Six-month antibody response to SARS-CoV-2 in healthcare workers assessed by virus neutralisation and commercial assays',
# 							'The disease-induced herd immunity level for Covid-19 is substantially lower than the classical herd immunity level',
# 					'Lipid droplets fuels SARS-CoV-2 replication and inflammatory response', 
# 					'RBD mutations from circulating SARS-CoV-2 strains enhance the structural 2 stability and human ACE2 affinity of the spike protein', 
# 					'Convalescent plasma is ineffective for covid-19']

# select_evidence(model, sentence, corpus)

def get_evidence_for_claim(claim, search_res):
	
	corpus = set()

	for sr in search_res[claim]:
		if not ".pdf" in sr: # do not process pdf
			new_sents, found_title = scrape_link(sr, claim)
			if found_title: # if we find exact source of the claim, no need to process anything more
				evidence = select_evidence(model, claim, new_sents)
				print(claim)
				print(evidence[0])
				print("+++"*100)
				return evidence
			else:
				for s in new_sents:
					corpus.add(s)

	evidence = select_evidence(model, claim, list(corpus))
	# print(claim)
	# print('CORPUS EVIDENCE')
	# print(evidence[0])
	# print("+++"*100)
	return evidence

tqdm.pandas()
matches['evidence'] = matches.progress_apply(lambda x: get_evidence_for_claim(x.claim, search_results5), axis=1)

#matches.to_csv('/content/drive/MyDrive/misinformation-NLP/tmp/siteMAtchEVIDENCE_LINKS_ADDITIONAL_1k.csv')

