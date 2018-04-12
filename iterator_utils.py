import collections
import tensorflow as tf

class BatchedInput(collections.namedtuple("BatchedInput",
                                           ("initializer",
                                            "input",
                                            "target",
                                            "input_sequence_length",
                                            "batch_size"))):
    pass


def get_iterator(input_dataset, output_dataset, input_vocab_table, batch_size, random_seed, pad,
                 input_max_len=None, output_buffer_size=None):
    """ Create batches for an input dataset along with a target dataset and get an iterator over the batches."""
    if not output_buffer_size: output_buffer_size = batch_size * 1000

    input_output_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    input_output_dataset = input_output_dataset.shuffle(output_buffer_size, random_seed)

    input_output_dataset = input_output_dataset.map(lambda inp, out: (tf.string_split([inp]).values, tf.string_to_number
    (tf.string_split([out]).values, tf.int32)))
    # remove input sequences of zero length.
    input_output_dataset = input_output_dataset.filter(lambda inp,out: tf.size(inp)>0)
    if input_max_len is not None:
        # truncate input sequences.
        input_output_dataset = input_output_dataset.map(lambda inp,out: (inp[:input_max_len],out))
    # Map words to ids
    input_output_dataset = input_output_dataset.map(lambda inp,out:
                                                    (tf.cast(input_vocab_table.lookup(inp),tf.int32),out))
    # get actual lengths of input sequences.
    # The sequences will be padded, so you need the actual lengths for later to know where the padding starts.
    input_output_dataset = input_output_dataset.map(lambda inp,out:(inp,out,tf.size(inp)))
    # get the index of the pad symbol from the vocabulary in order to pad the input sequences with that index.
    pad_id = tf.cast(input_vocab_table.lookup(tf.constant(pad)), tf.int32)
    def batching_func(x):
        # pad each batch to have input and output sequences of same length.
        return x.padded_batch(
            batch_size,
            padded_shapes = (tf.TensorShape([None]),
                           tf.TensorShape([None]),
                           tf.TensorShape([])),
            padding_values = (pad_id,0,0)
        )
    batched_dataset = batching_func(input_output_dataset)
    # create an iterator from the batched dataset.
    batched_iter = batched_dataset.make_initializable_iterator()
    (inputs,outputs,input_lens)=(batched_iter.get_next())
    return BatchedInput(initializer=batched_iter.initializer,
                        input=inputs, target=outputs,
                        input_sequence_length=input_lens,
                        batch_size=tf.size(input_lens))



def get_iterator_infer(input_dataset, input_vocab_table, batch_size, pad,
                 input_max_len=None):
    """ Create batches for an input dataset (when no target dataset is provided) and get an iterator over the batches."""
    pad_id = tf.cast(input_vocab_table.lookup(tf.constant(pad)),tf.int32)
    # split input sequences on space
    input_dataset = input_dataset.map(lambda inp: tf.string_split([inp]).values)
    # remove input sequences of zero length
    input_dataset = input_dataset.filter(lambda inp: tf.size(inp) > 0)
    if input_max_len:
        input_dataset = input_dataset.map(lambda inp: inp[:input_max_len])
    # Map words to ids
    input_dataset = input_dataset.map(lambda inp: tf.cast(input_vocab_table.lookup(inp), tf.int32))
    # get actual length of input sequence
    input_dataset = input_dataset.map(lambda inp: (inp, tf.size(inp)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]),
                           tf.TensorShape([])),
            padding_values=(pad_id, pad_id)
        )

    batched_dataset = batching_func(input_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (inputs, input_lens) = (batched_iter.get_next())
    return BatchedInput(initializer=batched_iter.initializer,
                        input=inputs,
                        target=None,
                        input_sequence_length=input_lens,
                        batch_size=tf.size(input_lens))