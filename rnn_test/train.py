#!/usr/bin/env python
# coding=utf-8

from __future__ import division

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import argparse
import time
import shutil
import pdb

import tensorflow as tf
from tensorflow.python.client import timeline

from model import BaseModel, DEBUG
from util import get_available_gpus, add_arguments, create_hparams

SINGLE_CARD_SPEED = None
WARM_UP_BATCH = 10


def make_config():
    config = tf.ConfigProto()

    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    config.intra_op_parallelism_threads = 0
    config.inter_op_parallelism_threads = 56

    return config


def train(model, config, hparams):
    sv = tf.train.Supervisor(
        is_chief=True,
        logdir="train_log",
        ready_for_local_init_op=None,
        local_init_op=model.local_var_init_op_group,
        saver=model.saver,
        global_step=model.global_step,
        summary_op=None,
        save_model_secs=600,
        summary_writer=None,
    )
    with sv.managed_session(
            master="", config=config, start_standard_services=False) as sess:

        tb_log_dir = "tblog"
        if os.path.exists(tb_log_dir):
            shutil.rmtree(tb_log_dir)
        else:
            os.mkdir(tb_log_dir)
        #merged_summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tb_log_dir, sess.graph)

        pass_id = 0
        batch_id = 0

        total_word_count = 0
        total_batch_id = 0
        start_time = None

        sess.run(model.iterator.initializer)

        while True:
            try:
                if DEBUG:
                    a, b, c, word_count, summaries = sess.run(
                        list(model.fetches.values()) + [model.word_count] +
                        [model.merged_summary_op])
                else:
                    a, b, word_count, summaries = sess.run(
                        list(model.fetches.values()) + [model.word_count] +
                        [model.merged_summary_op])
                if batch_id == WARM_UP_BATCH:
                    start_time = time.time()
                total_word_count += word_count if batch_id >= WARM_UP_BATCH else 0

                if batch_id and not batch_id % 5:
                    print("Pass %d, Batch %d " % (pass_id, batch_id),
                          " Loss: ", b)
                batch_id += 1
                total_batch_id += 1
                if batch_id == 5 and pass_id > 0:
                    model.saver.save(
                        sess,
                        "checkpoint/" + "model-%s.ckpt" % str(
                            pass_id
                        ),  # Suggestion: Conduct exp in /data/data1/v-qizhe/ to avoid disk space shortage
                        global_step=batch_id)
                writer.add_summary(summaries, total_batch_id)
                with open("train.log",'a') as file:
                    file.write(str(batch_id)+" "+str(b)+'\n')
            except tf.errors.OutOfRangeError:
                sess.run(model.iterator.initializer)
                batch_id = 0
                pass_id += 1
                continue


def eval(model, config, hparams):
    sv = tf.train.Supervisor(
        is_chief=True,
        logdir="eval_log",
        ready_for_local_init_op=None,
        local_init_op=model.local_var_init_op_group,
        saver=model.saver,
        global_step=model.global_step,
        summary_op=None,
        save_model_secs=600,
        summary_writer=None,
    )
    with sv.managed_session(
            master="", config=config, start_standard_services=False) as sess:

        pass_id = 0
        batch_id = 0

        total_word_count = 0
        start_time = None

        sess.run(model.iterator.initializer)
        ckpt = tf.train.get_checkpoint_state("checkpoint")
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            try:
                a, word_count = sess.run(
                    list(model.fetches.values()) + [model.word_count])

                if batch_id == WARM_UP_BATCH:
                    start_time = time.time()
                total_word_count += word_count if batch_id >= WARM_UP_BATCH else 0

                if batch_id and not batch_id % 5:
                    print("Pass %d, Batch %d " % (pass_id, batch_id),
                          " Loss: ", a)
                batch_id += 1
                with open("eval.log",'a') as file:
                    file.write(str(batch_id)+" "+str(a)+'\n')
            except tf.errors.OutOfRangeError:
                sess.run(model.iterator.initializer)
                batch_id = 0
                pass_id += 1
                continue


def infer(model, config, hparams):
    sv = tf.train.Supervisor(
        is_chief=True,
        logdir="eval_log",
        ready_for_local_init_op=None,
        local_init_op=model.local_var_init_op_group,
        saver=model.saver,
        global_step=model.global_step,
        summary_op=None,
        save_model_secs=600,
        summary_writer=None,
    )
    with sv.managed_session(
            master="", config=config, start_standard_services=False) as sess:

        pass_id = 0
        batch_id = 0

        total_word_count = 0
        start_time = None

        sess.run(model.iterator.initializer)
        ckpt = tf.train.get_checkpoint_state("checkpoint")
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            try:
                a, b, c, word_count = sess.run(
                    # list(model.fetches.values()) + [model.word_count]
                    [model.fetches["logits"]] + [model.fetches["answer"]] +
                    [model.fetches["seq_len"]] + [model.word_count])
                # the inference result is in the shape of [batch, seq-length, beam-width]
                #print(type(a), b, type(c))
                import numpy as np

                b = np.array(b)
                a = np.array(a)
                c = np.array(c)
                print(a.shape, b.shape, c.shape)
                with open("answer.txt", "a") as file:
                    for i in range(hparams.batch_size):
                        for j in range(c[0, i] - 1):
                            file.write(str(b[0, i, j]) + " ")
                        file.write("\n")
                with open("prediction.txt", "a") as file:
                    for i in range(hparams.batch_size):
                        for j in range(a.shape[1]):
                            file.write(str(a[i, j, 0]) + " ")
                        file.write("\n")
                # word_count = word_count[0]

                # print(logits.shape)

                if batch_id == WARM_UP_BATCH:
                    start_time = time.time()

                total_word_count += word_count if batch_id >= WARM_UP_BATCH else 0

                if batch_id and not batch_id % 5:
                    print("Pass %d, Batch %d " % (pass_id, batch_id))
                batch_id += 1
            except tf.errors.OutOfRangeError:
                print("Inference Done!")
                return
                sess.run(model.iterator.initializer)
                batch_id = 0
                pass_id += 1
                continue


def main(unused_argv):
    hparams = create_hparams(FLAGS)
    print(hparams)
    if hparams.mode == "train":
        _mode = tf.contrib.learn.ModeKeys.TRAIN
    elif hparams.mode == "eval":
        _mode = tf.contrib.learn.ModeKeys.EVAL
    elif hparams.mode == "infer":
        _mode = tf.contrib.learn.ModeKeys.INFER
    else:
        raise ("Unknown Mode!!!")

    model = BaseModel(hparams, mode=_mode)
    print("num_gpus = %d, batch size = %d" %
          (model.num_gpus, hparams.batch_size * model.num_gpus))
    config = make_config()
    if _mode == tf.contrib.learn.ModeKeys.TRAIN:
        train(model, config, hparams)
    if _mode == tf.contrib.learn.ModeKeys.EVAL:
        eval(model, config, hparams)
    if _mode == tf.contrib.learn.ModeKeys.INFER:
        infer(model, config, hparams)


if __name__ == "__main__":
    param_parser = argparse.ArgumentParser()
    add_arguments(param_parser)
    FLAGS, unparsed = param_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
