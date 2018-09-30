import tensorflow as tf

"""
n = tf.constant(5)

c = tf.TensorArray(tf.int32, n)
c = c.write(0,1)
c = c.write(1,1)

def cond(i,a,b,c0):
    return i<n

def body(i,a,b,c):
    c = c.write(i,a+b)
    return i+1, b, a+b, c

i,a,b,c = tf.while_loop(cond, body, (2,1,1,c))

c = c.stack()

print(tf.Session().run(c))
"""

O = tf.TensorArray(tf.int32, 3)
S = tf.TensorArray(tf.int32, 3)
X = tf.TensorArray(tf.int32, 3)

def cond(i, x, output, state):
    return i < 3

def body(i, x, output, state):
    x.write(i,[0,1,2,3])
    output.write(i,[0,0,0,0])
    state.write(i,[0,0,0,0])
    output.write(i, output.read(i) + state.read(i))
    state.write(i, x.read(i) + state.read(i))
    return i+1, x, output, state

i, xx, outputs, states = tf.while_loop(cond, body, (0, X, O, S))

outputs = outputs.stack()
states = states.stack()

print(tf.Session().run([outputs, states]))
