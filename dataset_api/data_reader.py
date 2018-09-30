import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

x = tf.placeholder(tf.float32, shape=[None,2])
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.batch(2)
data = np.random.sample((100,2))
print(data[:12,:])
iter = dataset.make_initializable_iterator() # create the iterator

with tf.Session() as sess:
    # feed the placeholder with data
    sess.run(iter.initializer, feed_dict={x: data})
    res_list = [[0,1,2], [0,1,2]]
    for i in range(3):
        for gpu_id in [6,7]:
            with tf.device('/gpu:%d'% gpu_id):
                res_list[gpu_id-6][i] = iter.get_next()
    print(sess.run(res_list))
