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
                 input_max_len=None, num_threads=4, output_buffer_size=None):
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



def get_iterator_infer(
        input_dataset, input_vocab_table, batch_size, input_max_len=None):
    # split source sentences on space
    input_dataset = input_dataset.map(lambda inp: tf.string_split([inp]).values)

    if input_max_len:
        input_dataset = input_dataset.map(lambda inp: inp[:input_max_len])
    # Convert words to ids
    input_dataset = input_dataset.map(
            lambda inp: tf.cast(input_vocab_table.lookup(inp),tf.int32))
    # Add in the seq lengths
    input_dataset = input_dataset.map(lambda inp: (inp,tf.size(inp)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([])),  # src_len
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(0,  # src
                            0))  # src_len -- unused

    # the input_dataset consists of tuples with se,seq_length
    # where seq_length is before the pudding.
    batched_dataset = batching_func(input_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (inputs,input_lens) = batched_iter.get_next()
    # by running in a session (inputs,input_lens)
    # I get the next element of the batched iterator
    return BatchedInput(initializer=batched_iter.initializer,
                        input=inputs, target=None,
                        input_sequence_length=input_lens,
                        batch_size=tf.size(input_lens))