#!/usr/bin/env python
import pprint
from utils import Data
import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
# Start session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# # Load Model
saver = tf.train.Saver()
saver.restore(sess, "./model.ckpt")
pred = model.y[0]
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_[0], model.y[0]))) # + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

batch_size = 1
num_samples = data.num_examples # 520
step_size = int(num_samples / batch_size) # 10

epochs = 1

output_size = 2
loss_value = np.zeros((len1*epochs, output_size))
real = np.zeros((len1*epochs, output_size))
predicted = np.zeros((len1*epochs, output_size))

for e in range(epochs):
    for i in range(step_size):
        batch = data.next_batch(batch_size)
        loss_value[len1*e + i] = loss.eval(feed_dict={model.x:batch[0], model.y_: batch[1], model.keep_prob: 1.0})
        real[len1*e + i] = batch[1]
        predicted[len1*e + i] = sess.run(pred, feed_dict={model.x:batch[0], model.y_: batch[1], model.keep_prob: 1.0})
        # if loss_value[len1*e + i] > 2:
        #     print('line in csv file: ', i+1)
        #     print("real", batch[1])
        #     print("predicted", sess.run(pred, feed_dict={model.x:batch[0], model.y_: batch[1], model.keep_prob: 1.0}))
        #     print('error', loss_value[len1*e + i])
# threshold = (loss_value.mean() + loss_value.max()) / 2
# bad = 0
# for i in range(loss_value.shape[0]):
#     if loss_value[i] > threshold:
#         bad += 1
#         print 'loss value: ' + str(loss_value[i][0]) + ' line number in csv file: ' + str(i+1)

print(data._X.shape) # (len1, 66, 200, 3)
print(data._y.shape) # (len1, 5)
# print 'number of -bad- predictions ' + str(bad)
# print 'threshold ' + str(threshold)
# print 'mean           ' + str(loss_value.mean())
# print 'min            ' + str(loss_value.min())
# print 'max            ' + str(loss_value.max())
# print 'std dev        ' + str(loss_value.std())
if output_size == 2:
    real = np.delete(real, 1, axis=1)
    predicted = np.delete(predicted, 1, axis=1)
t = np.arange(num_samples)
plt.plot(t,real,'b', label='real') 
plt.plot(t,predicted,'r', label='predicted') 
plt.legend()
track = 'Royal'
plt.title(track)
plt.show()
# plt.savefig(track + '.png')

mse = np.mean((real-predicted)**2)
print "MSE: "
print mse
# len2 = loss_value.shape[0]
# t = np.arange(len2)
# plt.plot(t,loss_value,'b', label='test') 
# plt.title('TEST LOSS')
# plt.show()
