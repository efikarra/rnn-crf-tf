
import collections
import tensorflow as tf

class BatchedInput(collections.namedtuple("BatchedInput",
                                           ("initializer",
                                            "input",
                                            "target",
                                            "input_sequence_length",
                                            "batch_size"))):
    pass

def get_iterator(input_dataset, output_dataset, input_vocab_table, batch_size, random_seed,
                 input_max_len, num_threads=4, output_buffer_size=None):
    if not output_buffer_size: output_buffer_size = batch_size * 1000
    input_output_dataset=tf.contrib.data.Dataset.zip((input_dataset, output_dataset))
    input_output_dataset = input_output_dataset.shuffle(output_buffer_size, random_seed)

    input_output_dataset=input_output_dataset.map(lambda inp,out: (tf.string_split([inp]).values,tf.string_to_number
                                                (tf.convert_to_tensor([out]),tf.int32)),
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
                           tf.TensorShape([None]),
                           tf.TensorShape([])),
            padding_values=(22,22,22)
        )
    batched_dataset = batching_func(input_output_dataset)
    # batched_dataset = input_output_dataset.batch(batch_size)
    batched_iter = batched_dataset.make_initializable_iterator()
    (inputs,outputs,input_lens)=(batched_iter.get_next())
    return BatchedInput(initializer=batched_iter.initializer,
                        input=inputs, target=outputs,
                        input_sequence_length=input_lens,
                        batch_size=tf.size(input_lens))

input_dataset = tf.contrib.data.TextLineDataset("data/train_input.txt")
UNK_ID = 0
input_vocab_table = tf.contrib.lookup.index_table_from_file("data/vocab.txt",default_value=UNK_ID)
output_dataset = tf.contrib.data.TextLineDataset("data/train_target.txt")
iterator = get_iterator(input_dataset, output_dataset, input_vocab_table, input_max_len=8, batch_size=2,
                 random_seed=None, num_threads=4, output_buffer_size=None)
with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)
    while True:
        try:
            res = sess.run([iterator.input,iterator.target,iterator.input_sequence_length,iterator.batch_size])
            print (res[0])
            print (res[1])
            print (res[2])
        except tf.errors.OutOfRangeError:
            break

