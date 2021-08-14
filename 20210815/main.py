import os
from tqdm import tqdm_notebook
import random
random.seed(1)
import warnings
warnings.simplefilter("ignore")

import numpy as np
from numba import njit

EPOCH = 5
BATCH_SIZE = 100
SEQ_LEN = 20
DATA_PATH = "./examples/word_language_model/data/wikitext-2/train.txt"
TOKENIZER_NAME = 'ProsusAI/finbert'

class DataSetLM : 
    
    def __init__(self, batch_size, seq_len, fixed_seq_len=False, shuffle=True) : 
        self._batch_size = batch_size
        self._seq_length = seq_len
        self._fixed_seq_len = fixed_seq_len
        self._shuffle = shuffle
        self._txt_loaded = False
        self._tok_loaded = False
        self._tokenized = False
        self._th = 0
        
    def load_txt(self, fname, min_word_cnt=5) : 
        assert fname.endswith(".txt"), 'it seems not from .txt file format'
        txt = []
        with open(fname, 'r') as f :
            for sent in f :
                txt.append(sent.strip())        
        self._filtered = list(filter(lambda x : len(x.split()) > 5, txt))
        print("data filtering : {} -> {}".format(len(txt), len(self._filtered)))
        self._txt_loaded = True
        
    def load_tokenizer(self, tokenizer, auto=False) : 
        if auto :
            if type(tokenizer) == str : 
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            else : 
                raise TypeError("input type should be str with auto=True")
        else : 
            assert hasattr(tokenizer, "_from_pretrained"), "if auto=False, you have to use pretrained hugingface's tokenizer"
            self._tokenizer = tokenizer
        self._tok_loaded = True
    
    def tokenize(self) : 
        assert self._txt_loaded, "load_txt first"
        assert self._tok_loaded, "load_tokenizer first"
                    
        print("tokenzation is started")
        self._wi = self._tokenizer(self._filtered, return_tensors='np')['input_ids']
        print("tokenzation is finished")
        self._tokenized = True
        self._encode(self._wi)
        
    def _resample(self) :
        if self._shuffle :             
            shuffled = np.random.choice(self._wi, self._wi.shape[0], False)
        return shuffled
    
    def _encode(self, filtered) :
        wi = np.concatenate(filtered)
        self._batched = _batchify(wi, self._batch_size)
        
    @property
    def vacab(self) : 
        assert self._tokenized, "load_tokenizer first"
        return self._tokenizer.vocab
    
    @property
    def iters(self) : 
        assert self._tokenized, "load_tokenizer first"
        if self._fixed_seq_len : 
            return self._batched.shape[0] - self._seq_length
        else : 
            return self._batched.shape[0] - 1
        
    def get_data(self) : 
        src, trg = _get_data(self._batched, self._th, self._seq_length)
        if trg.shape[0] == 1: # end of batch
            self._clear_th()
            if self._shuffle: 
                shuffled = self._resample()
                self._encode(shuffled)
        else : 
            self._step_th()
        return src, trg
    
    def _step_th(self) : 
        self._th += 1
        
    def _clear_th(self) : 
        self._th = 0
        
@njit
def _batchify(wi, batch_size) :
    seq_len = wi.shape[0]
    batch_length = seq_len//batch_size
    batch = wi[:batch_size*batch_length].reshape(batch_size, batch_length).transpose()
    return batch

@njit
def _get_data(batch, th, seq_len) : 
    nex = th+1
    remained = batch[nex:nex+seq_len].shape[0]
    length = min(seq_len, remained)
    return batch[th:th+length], batch[nex:nex+length]

if __name__ == "__main__" : 
    dataset = DataSetLM(BATCH_SIZE, SEQ_LEN)

    dataset.load_txt(DATA_PATH)
    dataset.load_tokenizer(tokenizer=TOKENIZER_NAME, auto=True)
    dataset.tokenize()
    
    for _ in range(EPOCH) :
        for _ in range(dataset.iters) : 
            src, trg = dataset.get_data()