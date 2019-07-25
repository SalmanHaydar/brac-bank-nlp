import os

abs_path = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(abs_path,"dataBank/tokenizer_GRU.pickle")
print(tokenizer_path)
for ind, keys in enumerate(['annual-fee','eligibility','facility','interest-rate','Unknown','required-documents']):

    print(str(ind)+" "+keys)