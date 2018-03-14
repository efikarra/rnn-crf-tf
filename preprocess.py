import nltk
import itertools
import numpy as np
import os
import argparse
import io

def build_vocabulary(tokenized_seqs, max_freq=0.0, min_freq=0.0):
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


def preprocess_data(params):
    data_folder = params.data_folder
    train_input_file = params.train_input_file
    train_target_file = params.train_target_file
    dev_input_file = params.dev_input_file
    dev_target_file = params.dev_target_file
    test_input_file = params.test_input_file
    test_target_file = params.test_target_file
    # data_folder = "experiments/data"
    # train_input_file="train_input_sgmt.txt"
    # train_target_file = "train_input_sgmt.txt"
    # dev_input_file = "val_input_sgmt.txt"
    # dev_target_file = "val_input_sgmt.txt"
    # test_input_file = "test_input_sgmt.txt"
    # test_target_file = "test_input_sgmt.txt"

    train_input=load_data_from_file(os.path.join(data_folder,train_input_file))
    train_target = load_data_from_file(os.path.join(data_folder,train_target_file))
    dev_input = load_data_from_file(os.path.join(data_folder,dev_input_file))
    dev_target = load_data_from_file(os.path.join(data_folder,dev_target_file))
    test_input = load_data_from_file(os.path.join(data_folder,test_input_file))
    test_target = load_data_from_file(os.path.join(data_folder,test_target_file))
    print("Train input: %d "% len(train_input))
    print("Train labels: %d " % len(train_input))
    print("Dev input: %d " % len(dev_input))
    print("Dev labels: %d " % len(dev_input))
    print("Test size: %d " % len(test_input))
    print("Test labels: %d " % len(test_input))

    train_tok=tokenize(train_input)
    dev_tok = tokenize(dev_input)
    test_tok = tokenize(test_input)

    avg_train_len,max_train_len,min_train_len=avg_seq_length(train_tok)
    avg_dev_len, max_dev_len, min_dev_len =avg_seq_length(dev_tok)
    avg_test_len, max_test_len, min_test_len =avg_seq_length(test_tok)
    print("Train: avg_seq_length = %.3f, max seq length = %d, min seq length = %d "%(avg_train_len,max_train_len,min_train_len))
    print("Dev: avg_seq_length = %.3f, max seq length = %d, min seq length = %d " % (avg_dev_len, max_dev_len, min_dev_len))
    print("Test: avg_seq_length = %.3f, max seq length = %d, min seq length = %d " % (avg_test_len, max_test_len, min_test_len))

    vocab = build_vocabulary(train_tok, max_freq=params.max_freq, min_freq=params.min_freq)
    print("Vocab size: %d " % len(vocab))
    save_to_file(os.path.join(data_folder,str(params.max_freq)+str(params.min_freq)+params.vocab_file), vocab)


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_folder", type=str, default=None)
    parser.add_argument("--train_input_file", type=str, default=None)
    parser.add_argument("--train_target_file", type=str, default=None)
    parser.add_argument("--dev_input_file", type=str, default=None)
    parser.add_argument("--dev_target_file", type=str, default=None)
    parser.add_argument("--test_input_file", type=str, default=None)
    parser.add_argument("--test_target_file", type=str, default=None)
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument("--min_freq", type=float, default=0.0)
    parser.add_argument("--max_freq", type=float, default=1.0)

def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    params, unparsed = parser.parse_known_args()
    preprocess_data(params)

if __name__ == '__main__':
    main()