def get_iterator(input_dataset, output_dataset, input_vocab_table, batch_size, random_seed,
                 input_max_len=None, num_threads=4, output_buffer_size=None):
    if not output_buffer_size: output_buffer_size = batch_size * 1000
    input_output_dataset=tf.contrib.data.Dataset.zip((input_dataset, output_dataset))
    input_output_dataset = input_output_dataset.shuffle(output_buffer_size, random_seed)

    input_output_dataset=input_output_dataset.map(lambda inp,out: (tf.string_split([inp]).values,tf.string_to_number
                                                (tf.convert_to_tensor(out),tf.int32)),
                                                  num_threads=num_threads, output_buffer_size=output_buffer_size)
    # remove input sequences of zero length
    input_output_dataset = input_output_dataset.filter(lambda inp,out: tf.size(inp)>0)
    if input_max_len:
        input_output_dataset = input_output_dataset.map(lambda inp,out: (inp[:input_max_len],out),
                                                        num_threads = num_threads,
                                                        output_buffer_size=output_buffer_size)
    # Map words to ids
    input_output_dataset = input_output_dataset.map(lambda inp,out:
                                                    (tf.cast(input_vocab_table.lookup(inp),tf.int32),out),
                                                    num_threads=num_threads, output_buffer_size=output_buffer_size)
    # get actual length of input sequence
    input_output_dataset = input_output_dataset.map(lambda inp,out:(inp,out,tf.size(inp)),
                                                    num_threads=num_threads, output_buffer_size=output_buffer_size)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]),
                           tf.TensorShape([]),
                           tf.TensorShape([])),
            padding_values=(22,22,22)
        )
    batched_dataset = batching_func(input_output_dataset)
    # batched_dataset = input_output_dataset.batch(batch_size)
    batched_iter = batched_dataset.make_initializable_iterator()
    (inputs,outputs,input_lens)=(batched_iter.get_next())
    return inputs,outputs,input_lens,batched_iter.initializer



import vocab_utils
import tensorflow as tf
input_vocab_table = vocab_utils.create_vocab_table("data/vocab.txt")
input_dataset = tf.contrib.data.TextLineDataset("data/train_input.txt")
output_dataset = tf.contrib.data.TextLineDataset("data/train_target.txt")
inputs,outputs,input_lens,initializer=get_iterator(input_dataset, output_dataset, input_vocab_table, 2, random_seed=None,
                 input_max_len=None, num_threads=4, output_buffer_size=None)

with tf.Session() as sess:
    sess.run(initializer)
    sess.run(tf.tables_initializer())
    inputs, outputs, input_lens=sess.run([inputs,outputs,input_lens])
    print inputs,outputs