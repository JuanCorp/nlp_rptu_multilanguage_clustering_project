from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfIdfGenerator(object):

    def __init__(self,texts,max_vocab=2000):
        self.max_vocab = max_vocab
        self.texts = texts

    def _load_transformer(self):
        self.transformer = TfidfVectorizer(max_features=self.max_vocab)


    def fit_tf_idf(self):
        self._load_transformer()
        self.transformer.fit(self.texts)
    

    

    def calculate_embeddings(self,texts):
        return self.transformer.transform(texts).toarray()
    

        
        