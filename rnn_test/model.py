from six.moves import xrange

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes

from util import *
from data_reader import *
import variable_mgr
import variable_mgr_util
from seq2seq_model import seq2seq_model
from BasicSeq2Seq import Seq2SeqModel

DEBUG = False


class BaseModel(object):
    def __init__(self,
                 hparams,
                 mode=tf.contrib.learn.ModeKeys.TRAIN,
                 worker_prefix=""):
        self.params = hparams
        self.num_gpus = len(get_available_gpus())
        self.cpu_device = "%s/cpu:0" % worker_prefix
        self.raw_devices = [
            "%s/%s:%i" % (worker_prefix, hparams.local_parameter_device, i)
            for i in xrange(self.num_gpus)
        ]

        self.iterator, self.tgt_bos_id, self.tgt_eos_id = self.get_input_iterator(
            hparams)
        self.word_count = tf.reduce_sum(
            self.iterator.source_sequence_length) + tf.reduce_sum(
                self.iterator.target_sequence_length)

        self.mode = mode
        self.model = Seq2SeqModel(mode, self.tgt_bos_id, self.tgt_eos_id)
        # self.model = seq2seq_model(mode, self.tgt_bos_id, self.tgt_eos_id)

        self.source = self.iterator.source
        self.target_input = self.iterator.target_input
        self.target_output = self.iterator.target_output
        self.source_sequence_length = self.iterator.source_sequence_length
        self.target_sequence_length = self.iterator.target_sequence_length

        self.batch_size = tf.reduce_sum(self.target_sequence_length)

        self.param_server_device = hparams.param_server_device
        self.local_parameter_device = hparams.local_parameter_device

        self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
        self.devices = self.variable_mgr.get_devices()
        self.global_step_device = self.cpu_device

        self.infer_helper = None
        self.debug_helper = None
        self.fetches = self.make_data_parallel(
            self.model.build_model,
            hparams=hparams,
            source=self.source,
            target_input=self.target_input,
            target_output=self.target_output,
            source_sequence_length=self.source_sequence_length,
            target_sequence_length=self.target_sequence_length,
        )
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN and DEBUG:
            #self.fetches["grads"] = self.debug_helper
            self.fetches["src_len"] = self.source_sequence_length
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.fetches["logits"] = self.infer_helper
            self.fetches["answer"] = self.target_output
            self.fetches["seq_len"] = self.target_sequence_length
        #fetches_list = nest.flatten(list(self.fetches.values()))
        #self.main_fetch_group = tf.group(*fetches_list)
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.merged_summary_op = tf.summary.merge_all()
        local_var_init_op = tf.local_variables_initializer()
        table_init_ops = tf.tables_initializer()
        variable_mgr_init_ops = [local_var_init_op]
        if table_init_ops:
            variable_mgr_init_ops.extend([table_init_ops])
        with tf.control_dependencies([local_var_init_op]):
            variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
        self.local_var_init_op_group = tf.group(*variable_mgr_init_ops)

        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

    def get_input_iterator(self, hparams):
        return get_iterator(
            src_file_name=hparams.src_file_name,
            tgt_file_name=hparams.tgt_file_name,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            batch_size=hparams.batch_size,
            num_splits=self.num_gpus,
            disable_shuffle=hparams.disable_shuffle,
            output_buffer_size=self.num_gpus * 1000 * hparams.batch_size,
            num_buckets=hparams.num_buckets,
        )

    def make_data_parallel(self, fn, **kwargs):
        """ Wrapper for data parallelism.
        """
        in_splits = {}
        for k, v in kwargs.items():
            if isinstance(v, list):  # input tensors
                in_splits[k] = v
            else:  # hyper parameters
                in_splits[k] = [v] * len(self.devices)

        losses = []
        device_grads = []
        for device_num in range(len(self.devices)):
            # when using PS mode, learnable parameters are placed on different
            # GPU devices. when using all-reduced algorithm, each GPU card has
            # an entire copy of model parameters.
            with self.variable_mgr.create_outer_variable_scope(
                    device_num), tf.name_scope(
                        "tower_%i" % device_num) as name_scope:
                results = self.add_forward_pass_and_gradients(
                    device_num, device_num,
                    **{k: v[device_num]
                       for k, v in in_splits.items()})
                if self.mode != tf.contrib.learn.ModeKeys.INFER:
                    losses.append(results["loss"])
                if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                    device_grads.append(results["gradvars"])
                if self.mode == tf.contrib.learn.ModeKeys.INFER:
                    self.infer_helper = results["logits"]

        with tf.device(self.global_step_device):
            self.global_step = tf.train.get_or_create_global_step()

        return self.build_gradient_merge_and_update(self.global_step, losses,
                                                    device_grads)

    def add_forward_pass_and_gradients(self, rel_device_num, abs_device_num,
                                       **inputs):
        """
        Args:
          rel_device_num: local worker device index.
          abs_device_num: global graph device index.
        """

        with tf.device(self.devices[rel_device_num]):
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                logits, loss, final_context_state = self.model.build_model(
                    source=inputs["source"],
                    target_input=inputs["target_input"],
                    target_output=inputs["target_output"],
                    source_sequence_length=inputs["source_sequence_length"],
                    target_sequence_length=inputs["target_sequence_length"],
                    hparams=inputs["hparams"],
                )
            if self.mode == tf.contrib.learn.ModeKeys.INFER:
                self.infer_helper, _, final_context_state = self.model.build_model(
                    source=inputs["source"],
                    target_input=inputs[
                        "target_input"],  # Should be None or invalid
                    target_output=inputs[
                        "target_output"],  # Should be None or invalid
                    source_sequence_length=inputs["source_sequence_length"],
                    target_sequence_length=inputs[
                        "target_sequence_length"],  # Should be None or invalid
                    hparams=inputs["hparams"],
                )

            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                params = self.variable_mgr.trainable_variables_on_device(
                    rel_device_num, abs_device_num)
                grads = tf.gradients(
                    loss,
                    params,
                    aggregation_method=tf.AggregationMethod.DEFAULT)
                grads, _ = tf.clip_by_global_norm(
                    grads, self.params.max_gradient_norm)
                gradvars = list(zip(grads, params))
                self.debug_helper = grads
                return {
                    "logits": logits,
                    "loss": loss,
                    "final_context_state": final_context_state,
                    "gradvars": gradvars,
                }
            if self.mode == tf.contrib.learn.ModeKeys.EVAL:
                return {
                    "logits": logits,
                    "loss": loss,
                    "final_context_state": final_context_state,
                }
            if self.mode == tf.contrib.learn.ModeKeys.INFER:
                return {
                    "logits": self.infer_helper,
                    "final_context_state": final_context_state
                }

    def build_gradient_merge_and_update(self, global_step, losses,
                                        device_grads):
        fetches = {}

        apply_gradient_devices = self.devices
        gradient_state = device_grads

        training_ops = []

        # gradient_state is the merged gradient.
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            apply_gradient_devices, gradient_state = self.variable_mgr.preprocess_device_grads(
                device_grads, self.params.independent_replica)

        for d, device in enumerate(apply_gradient_devices):
            with tf.device(device):
                if self.mode != tf.contrib.learn.ModeKeys.INFER:
                    average_loss = (losses[d]
                                    if self.params.independent_replica else
                                    tf.reduce_sum(losses))
                if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                    avg_grads = self.variable_mgr.get_gradients_to_apply(
                        d, gradient_state)
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                # add gradient clipping moved to add_forward_pass_and_gradients
                self.learning_rate = tf.constant(self.params.learning_rate)
                opt = tf.train.AdamOptimizer(self.learning_rate)
                loss_scale_params = variable_mgr_util.AutoLossScaleParams(
                    enable_auto_loss_scale=False,
                    loss_scale=None,
                    loss_scale_normal_steps=None,
                    inc_loss_scale_every_n=1000,
                    is_chief=True,
                )
                # append optimizer operators into the graph
                self.variable_mgr.append_apply_gradients_ops(
                    gradient_state, opt, avg_grads, training_ops,
                    loss_scale_params)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            fetches["train_op"] = tf.group(training_ops)
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            fetches["average_loss"] = (
                average_loss if self.params.independent_replica else
                average_loss / tf.to_float(self.batch_size))
        return fetches
