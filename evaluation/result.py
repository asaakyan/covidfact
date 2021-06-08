from fairseq.models.roberta import RobertaModel
import os
import numpy as np
from sklearn.metrics import classification_report
os.environ["CUDA_VISIBLE_DEVICES"]='2'

roberta = RobertaModel.from_pretrained(
    '/local/nlpswordfish/tuhin/covidfact-roberta/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='./RTE-covidfact-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
f = open('predictions.txt','w')
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
m = {'not_entailment': 0, 'entailment' : 1}
pred = []
gold = []
with open('./RTE-covidfact/test1.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        f.write(prediction_label+'\t'+target+'\n')
        pred.append(m[prediction_label])
        gold.append(m[target])
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
pred = np.asarray(pred)
gold = np.asarray(gold)
print(classification_report(gold, pred))
