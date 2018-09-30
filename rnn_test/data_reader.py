import sys
import os
import time
import collections
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.contrib.data.python.ops import prefetching_ops


class BatchedInput(
        collections.namedtuple(
            "BatchedInput",
            (
                "initializer",
                "source",
                "target_input",
                "target_output",
                "source_sequence_length",
                "target_sequence_length",
            ),
        )):
    pass


def create_iterator(
        src_file_name,
        tgt_file_name,
        src_vocab_file,
        tgt_vocab_file,
        batch_size,
        bos="<s>",
        eos="</s>",
        unk_id=0,
        src_max_len=None,
        tgt_max_len=None,
        num_parallel_calls=28,
        num_buckets=5,
        output_buffer_size=None,
        disable_shuffle=False,
        num_splits=1,
):
    def __get_word_dict(vocab_file_path, unk_id):
        return tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_file_path,
            key_column_index=0,
            default_value=unk_id)

    if not output_buffer_size:
        output_buffer_size = batch_size * 1000

    src_vocab_table = __get_word_dict(src_vocab_file, unk_id)
    tgt_vocab_table = __get_word_dict(tgt_vocab_file, unk_id)

    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    tgt_bos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(bos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_dataset = tf.data.TextLineDataset(src_file_name)
    tgt_dataset = tf.data.TextLineDataset(tgt_file_name)

    dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    if not disable_shuffle:
        dataset = dataset.shuffle(
            buffer_size=output_buffer_size, reshuffle_each_iteration=True)

    src_tgt_dataset = dataset.map(
        lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_parallel_calls,
    ).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0)
    )

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(output_buffer_size)

    # convert word string to word index
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.cast(src_vocab_table.lookup(src), tf.int32),
            tf.cast(tgt_vocab_table.lookup(tgt), tf.int32),
        ),
        num_parallel_calls=num_parallel_calls,
    ).prefetch(output_buffer_size)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            src,
            tf.concat(([tgt_bos_id], tgt), 0),
            tf.concat((tgt, [tgt_eos_id]), 0),
        ),
        num_parallel_calls=num_parallel_calls,
    ).prefetch(output_buffer_size)

    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src,
            tgt_in,
            tgt_out,
            tf.size(src),
            tf.size(tgt_in),
        ),
        num_parallel_calls=num_parallel_calls,
    ).prefetch(output_buffer_size)

    def __batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # tgt_in
                tf.TensorShape([None]),  # tgt_out
                tf.TensorShape([]),  # src_len
                tf.TensorShape([]),  # tgt_len
            ),
            padding_values=(src_eos_id, tgt_eos_id, tgt_eos_id, 0, 0),
        )

    if num_buckets > 1:

        def __key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
            if tgt_max_len:
                bucket_width = (tgt_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            bucket_id = tgt_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def __reduce_func(unused_key, windowed_data):
            return __batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=__key_func,
                reduce_func=__reduce_func,
                window_size=batch_size))
    else:
        batched_dataset = __batching_func(src_tgt_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    return batched_iter, tgt_bos_id, tgt_eos_id


def get_iterator(
        src_file_name,
        tgt_file_name,
        src_vocab_file,
        tgt_vocab_file,
        batch_size,
        bos="<s>",
        eos="</s>",
        unk_id=0,
        src_max_len=None,
        tgt_max_len=None,
        num_parallel_calls=28,
        num_buckets=5,
        output_buffer_size=None,
        disable_shuffle=False,
        num_splits=1,
):

    batched_iter, tgt_bos_id, tgt_eos_id = create_iterator(
        src_file_name,
        tgt_file_name,
        src_vocab_file,
        tgt_vocab_file,
        batch_size,
        bos,
        eos,
        unk_id,
        src_max_len,
        tgt_max_len,
        num_parallel_calls,
        num_buckets,
        output_buffer_size,
        disable_shuffle,
        num_splits,
    )
    src_ids = [[] for _ in range(num_splits)]
    tgt_input_ids = [[] for _ in range(num_splits)]
    tgt_output_ids = [[] for _ in range(num_splits)]
    src_seq_len = [[] for _ in range(num_splits)]
    tgt_seq_len = [[] for _ in range(num_splits)]
    for i in range(num_splits):
        (
            src_ids[i],
            tgt_input_ids[i],
            tgt_output_ids[i],
            src_seq_len[i],
            tgt_seq_len[i],
        ) = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_input_ids,
        target_output=tgt_output_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len,
    ), tgt_bos_id, tgt_eos_id


def test_data_reader(src_file_name, tgt_file_name, src_vocab_file,
                     tgt_vocab_file, num_splits, batch_size):
    iterator = get_iterator(
        src_file_name,
        tgt_file_name,
        src_vocab_file,
        tgt_vocab_file,
        batch_size,
        num_splits=num_splits,
    )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        for i in range(1000):
            try:
                print(sess.run(iterator))
            except:
                raise ("Exceptions in data reader UT!!!", time.time())


if __name__ == "__main__":
    test_data_reader("data/src.txt", "data/tgt.txt", "data/src_vocab.txt",
                     "data/tgt_vocab.txt", 2, 2 * 2)
