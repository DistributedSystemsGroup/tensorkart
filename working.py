import matplotlib.pyplot as plt
import numpy as np

train_loss = np.load('training_loss/train_loss.npy')
test_loss = np.load('test_loss/test_loss.npy')
len1 = test_loss.shape[0]
t = np.arange(len1)
plt.plot(t,train_loss,'b', label='train') 
plt.plot(t,test_loss,'r', label='test') 
plt.title('LOSS WITH MODEL 1')
plt.legend()
plt.show()
