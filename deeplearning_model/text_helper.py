import numpy as np
import re
import string
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
class textHelpers:
    '''
    Clean texts and turn them into series of numbers
    Args:
    train_data
    test_data
    '''
    def __init__(self, train_data, test_data):
        self._train_data = train_data
        self._test_data = test_data
        self._preprocess()
    
    
    def _preProcessor(self, s):
        #remove punctuation
        s = re.sub('['+string.punctuation+']', ' ', s)
        #remove digits
        s = re.sub('['+string.digits+']', ' ', s)
        #remove foreign characters
        s = re.sub('[^a-zA-Z]', ' ', s)
        #remove line ends
        s = re.sub('\n', ' ', s)
        #turn to lower case
        s = s.lower()
        s = re.sub('[ ]+',' ', s)
        s = s.rstrip()
        return s
    
    def _preprocess(self):
        '''Remove punctuations'''
        train_text = self._train_data
        test_text = self._test_data
        self._train_data = [self._preProcessor(item) for item in train_text]
        self._test_data = [self._preProcessor(item) for item in test_text]
        
    def _tfidf_vectorizer(self):
        ''''Vectorize news'''
        tfidfVectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_features=5000)
        X_train_tfidf = tfidfVectorizer.fit_transform(self._train_data)
        X_test_tfidf = tfidfVectorizer.transform(self._test_data)
        vocab_index_dict = tfidfVectorizer.vocabulary_
        return X_train_tfidf, X_test_tfidf, vocab_index_dict
    
    def tfidf_weight(self):
        '''Calculate TfIdf weights for each word within each news'''
        train_text_words, test_text_words = self._text2words()
        X_train_tfidf, X_test_tfidf, vocab_index_dict = self._tfidf_vectorizer()
        train_weights = []
        test_weights = []
        #Generate dicts for words and corresponding tfidf weights
        for i, text in enumerate(train_text_words):
            word_weight = []
            for word in text:
                try:
                    word_index = vocab_index_dict.get(word)
                    w = X_train_tfidf[i, word_index]
                    word_weight.append(w)
                except:
                    word_weight.append(0)
            train_weights.append(word_weight)
        for i, text in enumerate(test_text_words):
            word_weight = []
            for word in news:
                try:
                    word_index = vocab_index_dict.get(word)
                    w = X_test_tfidf[i, word_index]
                    word_weight.append(w)
                except:
                    word_weight.append(0)
            test_weights.append(word_weight)      
        return train_weights, test_weights
    
    def _text2words(self):
        #Split each news into words
        train_text_words = []
        test_text_words = []
        for text in self._train_data:
           #Collect words for each news
           train_text_words.append(text.split())
        for text in self._test_data:
            test_text_words.append(text.split())
        return train_text_words, test_text_words
    
    def buildVocab(self):
        words = []
        for text in self._train_data:
           #Collect all the chars
           words.extend(text.split())
        #Calculate frequencies of each character
        word_freq = Counter(words)
        #Filter out those low frequency characters
        vocab = [u for u,v in word_freq.items() if v>3]
        if 'UNK' not in vocab:
            vocab.append('UNK')
        #Map each char into an ID
        word_id_map = dict(zip(vocab, range(len(vocab))))
        #Map each ID into a word
        id_word_map = dict(zip(word_id_map.values(), word_id_map.keys()))
        return vocab, word_id_map, id_word_map
    
    def text2vecs(self):
        #Map each word into an ID
        train_text_words, test_text_words = self._text2words()
        vocab, word_id_map, id_word_mapp = self.buildVocab()
        def word2id(c):
            try:
               ID = word_id_map[c]
            except:#Trun those less frequent words into UNK
               ID = word_id_map['UNK']
            return ID
        #Turn each news into a list of word Ids
        words_vecs = lambda words: [word2id(w) for w in words]
        train_text_vecs = [words_vecs(words) for words in train_text_words]
        test_text_vecs = [words_vecs(words) for words in test_text_words]
        return train_text_vecs, test_text_vecs