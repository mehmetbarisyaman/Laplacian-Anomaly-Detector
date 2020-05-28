import numpy as np
import math

def bsxMinus(X, X_transpose):
	out = np.zeros((len(X), len(X)))
	for i in range(len(X)):
		for j in range(len(X)):
			out[i][j] = X[i] - X_transpose[j]
	return out


def cauchy_normalize(X, a):
	out = np.zeros((len(X), len(X)))
	for i in range(len(X)):
		for j in range(len(X)):
			out[i][j] = 1/(1+pow((X[i][j]/a),2))
	return out

def partial_correlation_normalize(X, z_size):
  M = X.mean(0)
  C = np.cov(X)
  print(C)
  Q = np.linalg.inv(C)
  A = np.zeros((z_size, z_size))
  for i in range(z_size-1):
    for j in range(i+1, z_size):
      w = abs(Q[i,j]/math.sqrt(Q[i,i]*Q[j,j]))
      A[i,j] = w
      A[j,i] = w
  return A


def pow_minus_1_over_2(X):
	out = np.zeros((len(X), len(X)))
	for i in range(len(X)):
		for j in range(len(X)):
			out[i][j] = pow(X[i][j], -0.5)
	return out


def lad_cauchy(X):
  image_size = X.shape
  X = X.reshape(image_size[0]*image_size[1], image_size[2])
  M = X.mean(0)
  out = np.zeros((image_size[0]* image_size[1], 1))

  A = np.absolute(bsxMinus(M, M.transpose()))
  a = np.mean(M)
  #A = cauchy_normalize(A, a)
  A = partial_correlation_normalize(A, image_size[2])
  A = np.subtract(A, np.eye(len(A), len(A[0])))

  D = np.diag(A.sum(axis=0))
  L = np.subtract(D , A)
  	
  L = np.multiply(pow_minus_1_over_2(D), L, pow_minus_1_over_2(D))

  for j in range(image_size[0]*image_size[1]):
  	x = X[j]
  	out[j] = np.subtract(x, M).dot(L).dot(np.subtract(x, M).transpose())

  out = out.reshape(image_size[0], image_size[1])
  print(out)


data = np.arange(24).reshape(2, 3, 4)

lad_cauchy(data)







