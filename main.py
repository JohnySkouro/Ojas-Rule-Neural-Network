import numpy as np

def ojas(X, learning_rate = 0.1, iters = 100, features = 3):
    #create weights array initialized to random values
    w = np.random.rand(1,features)
    #w = np.array([0.5,0.5])
    #w = w.reshape(-1,1)

    # Init W
    W = np.copy(w)

    # Train - Fit
    for i in range(1,50):
        y = np.dot(X,w.T)
        delta_w = (np.multiply(y,X)) - (learning_rate * np.dot((np.square(y)),w))
        w = w + np.sum(delta_w)
    return w