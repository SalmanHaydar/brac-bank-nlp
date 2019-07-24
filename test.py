import os

abs_path = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(abs_path,"dataBank/tokenizer_GRU.pickle")
print(tokenizer_path)