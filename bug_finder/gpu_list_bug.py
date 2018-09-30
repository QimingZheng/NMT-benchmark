#!/usr/bin/env python
#coding=utf-8
import tensorflow as tf
from tensorflow.python.client import device_lib

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def get_available_gpus():
    """Returns a list of available GPU devices names.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]

def main():
    gpu_list = get_available_gpus()
    print(gpu_list)
    print("=================")
    for i in gpu_list:
        print(i)
    with tf.device(gpu_list[0]):
        a = tf.Variable([1,2,3])
        b = tf.Variable([1,2,3])
        c = a+b
        with tf.Session() as sess:
            sess.run(c)
            print("hello from", gpu_list[0])

main()
