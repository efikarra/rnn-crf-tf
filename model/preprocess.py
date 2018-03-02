import nltk
import itertools
import numpy as np
import os
import io

def build_vocabulary(tokenized_seqs, max_freq=1.0, min_freq=0.0, stopwords=None):
    # compute word frequencies
    vocab=set()
    freq_dist=nltk.FreqDist(itertools.chain(*tokenized_seqs))
    sorted_words = dict(sorted(freq_dist.items(), key=lambda x: x[1])).keys()
    max_idx=len(vocab)-1
    min_idx=0
    if isinstance(max_freq, float):
        max_idx=len(sorted_words)-int(np.ceil(max_freq * len(sorted_words)))
    if isinstance(min_freq, float):
        min_idx=int(np.ceil(min_freq * len(sorted_words)))
    vocab=sorted_words[min_idx:max_idx]
    return vocab

def tokenize(data, delimiter=" "):
    data_tok=[]
    for d in data:
        data_tok.append(d.split(delimiter))
    return data_tok


def load_data_from_file(filepath):
    with io.open(filepath, "r", encoding="utf-8") as f:
        data=f.read().splitlines()
    return data


def save_to_file(filepath, data):
    with io.open(filepath, 'w', encoding="utf-8") as f:
        newline = ""
        for d in data:
            f.write(unicode(newline+str(d)))
            newline="\n"


def avg_seq_length(seq_tok):
    sum_len=0.0
    max=0.0
    min=10000000
    for seq in seq_tok:
        sum_len+=len(seq)
        if len(seq)>max:max=len(seq)
        if len(seq) < min: min = len(seq)
    return sum_len/len(seq_tok),max,min


if __name__=="__main__":
    data_folder = "data"
    train_input_file="train_input.txt"
    train_target_file = "train_input.txt"
    dev_input_file = "dev_input.txt"
    dev_target_file = "dev_input.txt"
    test_input_file = "test_input.txt"
    test_target_file = "test_input.txt"

    train_input=load_data_from_file(os.path.join(data_folder,train_input_file))
    train_target = load_data_from_file(os.path.join(data_folder,train_target_file))
    dev_input = load_data_from_file(os.path.join(data_folder,dev_input_file))
    dev_target = load_data_from_file(os.path.join(data_folder,dev_target_file))
    test_input = load_data_from_file(os.path.join(data_folder,test_input_file))
    test_target = load_data_from_file(os.path.join(data_folder,test_target_file))
    print("Train size: %d "% len(train_input))
    print("Dev size: %d " % len(dev_input))
    print("Test size: %d " % len(test_input))

    train_tok=tokenize(train_input)
    dev_tok = tokenize(dev_input)
    test_tok = tokenize(test_input)

    avg_train_len,max_train_len,min_train_len=avg_seq_length(train_tok)
    avg_dev_len, max_dev_len, min_dev_len =avg_seq_length(dev_tok)
    avg_test_len, max_test_len, min_test_len =avg_seq_length(test_tok)
    print("Train: avg_seq_length = %.3f, max seq length = %d, min seq length = %d "%(avg_train_len,max_train_len,min_train_len))
    print("Dev: avg_seq_length = %.3f, max seq length = %d, min seq length = %d " % (avg_dev_len, max_dev_len, min_dev_len))
    print("Test: avg_seq_length = %.3f, max seq length = %d, min seq length = %d " % (avg_test_len, max_test_len, min_test_len))

    vocab = build_vocabulary(train_tok, max_freq=0.2)
    print("Vocab size: %d " % len(vocab))
    save_to_file("data/vocab.txt", vocab)
