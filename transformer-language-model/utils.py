import re
from collections import Counter 
import numpy as np
import torch
import sentencepiece as spm

def clean_text(text):
    text = re.sub(r'[^(\x00-\x7F)]+', ' ', text)
    text = re.sub(' +', ' ', text)
    text = re.sub(r'(\n\s+)', '\n\n', text)
    text = re.sub(r'\x0c', '', text)
    text = re.sub(r'\|', '', text)
    text = re.sub('%', '', text)
    text = re.sub('~', '', text)
    return text

def generate_data(text_as_int, batch_size, seq_len):
    max_len = len(text_as_int) % (batch_size * seq_len)
    Xs = torch.from_numpy(text_as_int[:-max_len]).view(-1, batch_size, seq_len)
    Ys = torch.from_numpy(text_as_int[1:-max_len + 1]).view(-1, batch_size, seq_len)
    assert Xs.size() == Ys.size(), 'dim mismatch'

    for i in range(Xs.size(0)):
        yield Xs[i], Ys[i]

