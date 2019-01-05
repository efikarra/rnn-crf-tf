This repository implements a Recurrent Neural Network with a Conditional Random Field layer (RNN-CRF model) for word sequence tagging.
You can select the GPU on which you want to run the model by setting the paramter "gpu" to the corresponding GPU number.
Input data: 
1. Text file including your input word sequences (one sequence per line and words are separated by space).
2. Text file including the corresponding label ids sequences (one sequence per line and labels are separated by space). The labels should not be text names, they should have been mapped into ids.
3. Vocabulary text file (one word per line).

You can find example input data in the folder "data".
This project supports creation of word embeddings for your input vocabulary. When running the model training, set the parameter create_new_embeddings to True and point the parameter embedding_path to the pre-trained embeddings file in your machine. Also, set the parameter emb_size equal to the size of your pre-trained embeddings.
You can download pre-trained embeddings here: https://github.com/stanfordnlp/GloVe
The code saves regular checkpoints in the folder out_dir.

The code is written in Python 2 with Tensorflow (tested for version 1.8).

<h3> Run the code </h3>
To train the model, run the following command:

python main.py --train_input_path=../data/train_word_seq.txt --train_target_path=../data/train_label_seq.txt 
--val_input_path=../data/dev_word_seq.txt --val_target_path=../data/dev_label_seq.txt 
--input_emb_file=../data/embedding.txt --vocab_path=../data/train_vocab.txt 
--embedding_path=../data/embeddings/glove.840B.300d.txt --out_dir=../models --create_new_embeddings=True 
--model_architecture=simple_rnn --num_units=64 --num_layers=1 --n_classes=21 --unit_type=rnn 
--in_to_hidden_dropout=0.1 --num_ckpt_epochs=1 --optimizer=adam --learning_rate=0.01 --emb_size=300 
--batch_size=64 --num_epochs=5 --input_emb_trainable=True --num_ckpt_epochs=1 --gpu=1 > log.txt 2>&1 &

To evaluate the model on a test dataset, run the following command:

python2 main.py --eval_input_path=../data/test_word_seq.txt --eval_target_path=../data/test_label_seq.txt 
--input_emb_file=../data/embedding.txt --embedding_path=../data/embeddings/glove.840B.300d.txt 
--out_dir=../results --create_new_embeddings=False --vocab_path=../data/train_vocab.txt 
--eval_output_folder=../models --eval_batch_size=64 --create_new_embeddings=False --model_architecture=simple_rnn 
--num_units=64 --init_weight=0.1 --num_layers=1 --n_classes=21 --unit_type=rnn --in_to_hidden_dropout=0.1 
--num_ckpt_epochs=1 --learning_rate=0.01 --emb_size=300 --batch_size=64 --num_epochs=5 
--eval_batch_size=4 --input_emb_trainable=True --num_ckpt_epochs=1 --gpu=1 > log.txt 2>&1 &

