import numpy as np

np.random.seed(444)

N = 10000
sigma = 0.1
noise = sigma * np.random.randn(N)
x = np.linspace(0, 2, N)
d = 3 + 2 * x + noise
d.shape = (N, 1)

# We need to prepend a column vector of 1s to `x`.
X = np.column_stack((np.ones(N, dtype=x.dtype), x))
print(X.shape)

#---------

Xplus = np.linalg.pinv(X)
w_opt = Xplus @ d
print(w_opt)

#---------

import itertools as it

def py_descent(x, d, mu, N_epochs):
    N = len(x)
    f = 2 / N

    # "Empty" predictions, errors, weights, gradients.
    y = [0] * N
    w = [0, 0]
    grad = [0, 0]

    for _ in it.repeat(None, N_epochs):
        # Can't use a generator because we need to
        # access its elements twice.
        err = tuple(i - j for i, j in zip(d, y))
        grad[0] = f * sum(err)
        grad[1] = f * sum(i * j for i, j in zip(err, x))
        w = [i + mu * j for i, j in zip(w, grad)]
        y = (w[0] + w[1] * i for i in x)
    return w

#---------

import time

x_list = x.tolist()
d_list = d.squeeze().tolist()  # Need 1d lists

# `mu` is a step size, or scaling factor.
mu = 0.001
N_epochs = 10000

t0 = time.time()
py_w = py_descent(x_list, d_list, mu, N_epochs)
t1 = time.time()

print(py_w)

print('Solve time: {:.2f} seconds'.format(round(t1 - t0, 2)))

#---------

def np_descent(x, d, mu, N_epochs):
    d = d.squeeze()
    N = len(x)
    f = 2 / N

    y = np.zeros(N)
    err = np.zeros(N)
    w = np.zeros(2)
    grad = np.empty(2)

    for _ in it.repeat(None, N_epochs):
        np.subtract(d, y, out=err)
        grad[:] = f * np.sum(err), f * (err @ x)
        w = w + mu * grad
        y = w[0] + w[1] * x
    return w

np_w = np_descent(x, d, mu, N_epochs)
print(np_w)

#---------

import timeit

setup = ("from __main__ import x, d, mu, N_epochs, np_descent;"
         "import numpy as np")
repeat = 5
number = 5  # Number of loops within each repeat

np_times = timeit.repeat('np_descent(x, d, mu, N_epochs)', setup=setup,
                         repeat=repeat, number=number)

print(min(np_times) / number)
#---------

import tensorflow as tf

def tf_descent(X_tf, d_tf, mu, N_epochs):
    N = X_tf.get_shape().as_list()[0]
    f = 2 / N

    w = tf.Variable(tf.zeros((2, 1)), name="w_tf")
    y = tf.matmul(X_tf, w, name="y_tf")
    e = y - d_tf
    grad = f * tf.matmul(tf.transpose(X_tf), e)

    training_op = tf.assign(w, w - mu * grad)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        for epoch in range(N_epochs):
            sess.run(training_op)
        opt = w.eval()
    return opt

X_tf = tf.constant(X, dtype=tf.float32, name="X_tf")
d_tf = tf.constant(d, dtype=tf.float32, name="d_tf")

tf_w = tf_descent(X_tf, d_tf, mu, N_epochs)
print(tf_w)

#---------

type(X_tf)

#---------

setup = ("from __main__ import X_tf, d_tf, mu, N_epochs, tf_descent;"
         "import tensorflow as tf")

tf_times = timeit.repeat("tf_descent(X_tf, d_tf, mu, N_epochs)", setup=setup,
                         repeat=repeat, number=number)

print(min(tf_times) / number)
