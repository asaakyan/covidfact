from transformers import pipeline
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
import pandas as pd

fname = '' #'/content/drive/MyDrive/misinformation-NLP/results/additional_1k.csv'
output_fname = ''
claims = pd.read_csv(fname)

tokenizer = RobertaTokenizerFast.from_pretrained("amoux/roberta-cord19-1M7k")
model = RobertaForMaskedLM.from_pretrained("amoux/roberta-cord19-1M7k")

fillmask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

#dir = '/content/drive/MyDrive/misinformation-NLP/results/'
data = []
i=0
for claim, keyword, source, flair in tqdm(zip(claims['claim'], 
													claims['keyword'], 
													claims['src_url'],
													claims['flair'])):
	i+=1
	if claim[-1] != ".":
		# if there is no . at the end and we are replacing last word, the model will just predict a punctuation mark
		claim = claim +  "."

	masked_text = claim.lower().replace(keyword, tokenizer.mask_token)
	#print(masked_text)

	try:
		suggs = fillmask(masked_text, top_k=10) #CHANGED TO 10
		suggs = [sent['sequence'].replace("<s>", "").replace("</s>", "") for sent in suggs]
		suggs = set(suggs)
	except:
		pass

	for sug in suggs:
		if sug != claim.lower():
		  data.append([claim, keyword, sug, source, flair])


all_fakes = pd.DataFrame(data, columns=['claim', 'keyword', 'false_claim', 'source', 'flair'])
all_fakes.to_csv(dir+output_fname, index=False)
