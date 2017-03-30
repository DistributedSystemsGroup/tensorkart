#!/usr/bin/env python
import pprint
from utils import Data
import model
import tensorflow as tf
import numpy as np

def amp(arr):
    steer = arr[0][0]
    threshold = 0.3
    gain = 1.2
    # threshold = 0.4
    # gain = 1.9
    if steer > threshold:
        steer = max(gain*steer, 1)
    elif steer < -threshold:
        steer = max(gain*steer, -1)
    # else:
    #     steer *= 0.5
    arr[0][0] = steer
    return arr

# Load Test Data
data = Data(1)
len1 = data._y.shape[0]
# print(data._X.shape) # (len1, 66, 200, 3)
# print(data._y.shape) # (len1, 5)
# Start session
sess = tf.InteractiveSession()

# # Load Model
saver = tf.train.Saver()
saver.restore(sess, "./model.ckpt")

###### first method #######
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_[0], model.y[0]))) # + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

sess.run(tf.global_variables_initializer())

batch_size = 1
num_samples = data.num_examples # 520
step_size = int(num_samples / batch_size) # 10

epochs = 10

loss_value = np.zeros((len1*epochs, 1))

for e in range(epochs):
	for i in range(step_size):
	    batch = data.next_batch(batch_size)
	    loss_value[len1*e + i] = loss.eval(feed_dict={model.x:batch[0], model.y_: batch[1], model.keep_prob: 1.0})
    # if i%10 == 0:
    # print("step: %d loss: %g"%(batch_size + i, loss_value))
# pprint.pprint(loss_value/loss_value.mean())
print 'mean           ' + str(loss_value.mean())
print 'min            ' + str(loss_value.min())
print 'max            ' + str(loss_value.max())
print 'std dev        ' + str(loss_value.std())

