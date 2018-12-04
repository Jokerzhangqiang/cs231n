import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
                        简单实现
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
   y
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
      score = np.matmul(X[i] , W)# y =x*w 
      score = np.max(0,score)
      score = np.exp(score)#取对数
      softmax_sum = np.sum(score)#求和得到Sj的分母
      score /=softmax_sum#除以分母得到Sj
      for j in range(W.shape[1]):#计算梯度
          if j != y[i]:#这个不太懂，为什么判断
              dW[:,j] += score[j] * X[i]
          else:
              dW[:,j] -=(1 - score[j]) * X[i]
      loss -= np.log(score[y[i]])#计算交叉熵
  loss /=X.shape[0]
  dW /= X.shape[0]
  loss += reg * np.sum(W * W)#加上正则项
  dW += 2* reg * W 
      
      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
                         向量版本
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = np.matmul(X,W)
  
  score = np.exp(score)
  score /=np.sum(score,axis = 1,keepdims = True)#计算softmax
  ds = np.copy(score) #初始化loss对score的梯度
  ds[np.arange(X.shape[0]),y] -= 1 #求出score的梯度
  dW = np.dot(X.T, ds)
  loss = score[np.arange(X.shape[0]) , y ]
  loss = -np.log(loss).sum()
  loss /= X.shape[0]
  dW /=X.shape[0]
  loss += reg * np.sum(W*W)
  dW += 2*reg *W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

