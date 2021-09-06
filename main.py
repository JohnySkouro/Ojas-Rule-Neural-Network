import numpy as np

def ojas(X, learning_rate = 0.1, iters = 100, features = 2):
    #create weights array initialized to random values
    w = np.random.rand(1,features)
    #w = np.array([0.5,0.5])
    #w = w.reshape(-1,1)

    # Init W
    W = np.copy(w)

    # Train - Fit
    for i in range(1,50):
        y = np.dot(X,w.T)
        delta_w = (learning_rate * np.multiply(y,X)) - (learning_rate * np.dot((np.square(y)),w))
        #delta_w = learning_rate * np.sum(np.multiply(y,X) - np.dot((np.square(y)),w))
        w = w + np.sum(delta_w)
    return w

def sangers(X, learning_rate = 0.1, iters = 100, features = 2):
    #create weights array initialized to random values
    w = np.random.rand(1,features)
    #w = np.array([0.5,0.5])
    #w = w.reshape(-1,1)
    W = np.copy(w)
    for i in range(1,50):
        y = np.dot(X,w.T)
        delta_w = learning_rate*(np.multiply(y,X) - np.dot(np.tril(np.outer(y,y.T)),w))
        w = w + np.sum(delta_w)
    return w
#TEST OJAS Rule
# Create Data
x = 0.5 * np.random.normal(size=10)
y = 0.5 * np.random.normal(size=10)
X = np.array([x , y]).T
X.shape

W = ojas(X)
print(W)