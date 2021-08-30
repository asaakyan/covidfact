# COVID-Fact

This repository contains data and code for our ACL 2021 paper 

                  COVID-Fact: Fact Extraction and Verification of Real-World Claims on COVID-19 Pandemic
                  Arkadiy Saakyan, Tuhin Chakrabarty, and Smaranda Muresan.

The full dataset is contained in the file COVIDFACT_dataset.jsonl. Every line is a dictionary with the following format below:

{"claim": "claim text", "label": "REFUTED or SUPPORTED", "evidence": [list of evidence sentences], "gold_source": "link", "flair": "post's flair"}

# Evidence Selection
The script used for evidence selection / claim filtration can be found in the folder doc_selection. There is also the script to select evidence for a particular claim. 

# Counter-claim Generation
The scripts used for counter-claim generation can be found in the folder contrastive_gen. get_attn_model.py is used to find the best head and layer of a fine-tuned BERT uncased model. get_top_words.py is used to obtain the salient words for a particular set of claims. Finally, gen_contrast_claims.py is used to replace salient words by outputs generated by RoBERTA fine-tuned on CORD-19.

You can find the BERT model fine-tuned on SciFact used to get the attention weights here: https://drive.google.com/drive/folders/1EH2nk3NLfcNAdyesPGu9u0_7altoTRXS?usp=sharing

# Evaluation
The folder evaluation provides the scripts needed to recreate Table 4 and Table 5 of the paper. eval_data folder provides the dataset already processed in the format necessary for the experiments.

Download the fairseq folder from 
              https://drive.google.com/file/d/1WzDrE3DQHLnlM6j_nokFP5D9ZxAB2f66/view?usp=sharing 
and use roberta_train.sh 


# Citation
                @inproceedings{saakyan-etal-2021-covid,
                    title = "{COVID}-Fact: Fact Extraction and Verification of Real-World Claims on {COVID}-19 Pandemic",
                    author = "Saakyan, Arkadiy  and
                      Chakrabarty, Tuhin  and
                      Muresan, Smaranda",
                    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
                    month = aug,
                    year = "2021",
                    address = "Online",
                    publisher = "Association for Computational Linguistics",
                    url = "https://aclanthology.org/2021.acl-long.165",
                    pages = "2116--2129",
                }


# Contact
a.saakyan@columbia.edu <br>
tuhin.chakr@cs.columbia.edu

