import matplotlib.pyplot as plt
import numpy as np

train_loss = np.load('training_loss/train_loss.npy')
test_loss = np.load('test_loss/test_loss.npy')
print 'min test: ' +  str(np.min(test_loss))
print 'min train: ' + str(np.min(train_loss))
len1 = test_loss.shape[0]
t = np.arange(len1)
plt.plot(t,train_loss,'b', label='train') 
plt.plot(t,test_loss,'r', label='test') 
# plt.plot(t,np.absolute(np.subtract(test_loss, train_loss)),'r', label='test') 

plt.title('LOSS')
plt.grid()
plt.legend()
plt.show()
# plt.savefig('loss_plot.png')

