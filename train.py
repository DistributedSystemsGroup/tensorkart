#!/usr/bin/env python

from utils import Data
import model
import tensorflow as tf
import numpy as np
# Load Training Data
train_data = Data()
test_data = Data(1)
# Start session
sess = tf.InteractiveSession()

# Learning Functions
L2NormConst = 0.001
train_vars = tf.trainable_variables()
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess.run(tf.global_variables_initializer())

# Training loop variables
epochs = 10
batch_size = 50
num_samples = test_data.num_examples
print num_samples
step_size = int(num_samples / batch_size) # 11
print step_size
train_loss_values = []
test_loss_values = []
for epoch in range(epochs):
    for i in range(step_size):
    	count = epoch*step_size + i
        train_batch = train_data.next_batch(batch_size)
        test_batch = test_data.next_batch(batch_size)
        train_step.run(feed_dict={model.x: train_batch[0], model.y_: train_batch[1], model.keep_prob: 0.8})

        if count%10 == 0:
          train_loss_value = loss.eval(feed_dict={model.x:train_batch[0], model.y_: train_batch[1], model.keep_prob: 1.0})
          test_loss_value = loss.eval(feed_dict={model.x:test_batch[0], model.y_: test_batch[1], model.keep_prob: 1.0})
          print("epoch: %d step: %d training loss: %g"%(epoch, epoch * batch_size + i, train_loss_value))
          print("epoch: %d step: %d test loss: %g"%(epoch, epoch * batch_size + i, test_loss_value))
          print('-----------')
          train_loss_values.append(train_loss_value)
          test_loss_values.append(test_loss_value)
# Save the Model
saver = tf.train.Saver()
np_train_loss_values = np.array(train_loss_values)
np_test_loss_values = np.array(test_loss_values)
saver.save(sess, "model.ckpt")
np.save('training_loss/train_loss', np_train_loss_values)
np.save('test_loss/test_loss', np_test_loss_values)
