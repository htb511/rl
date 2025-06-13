import numpy as np
import matplotlib.pyplot as plt

# Known data points
X = np.array([-2, -1, 1, 2], dtype=np.float32)
Y = np.array([6, 4, 0, -2], dtype=np.float32)

# model prediction
def forward(w, x, b):
    return w * x + b

# MSE
def loss(y, y_predicted):
    return ((y - y_predicted)**2).mean()

# gradient of loss wrt weight
def gradient_dl_dw(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

def gradient_dl_db(x, y, y_predicted):
    return np.dot(2, y_predicted-y).mean()

# Training
learning_rate = 0.01
epochs = 500

w = 10
b = 10
for epoch in range(epochs):
    # forward pass
    # calculate predictions
    y_predicted = forward(w, X, b)

    # calculate losses
    l = loss(Y, y_predicted)

    # backpropagation
    # calculate gradients
    dw = gradient_dl_dw(X,Y, y_predicted)

    db = gradient_dl_db(X, Y, y_predicted)

    # gradient descent
    # update weights
    w -= learning_rate * dw

    b -= learning_rate * db

    # print info
    print(f'epoch {epoch+1}: w={w:.3f}, b={b:.3f}, loss={l:0.8f}, dw={dw:.3f}, forward(10)={forward(w,10,b):0.3f}')

plt.scatter(X, Y)
plt.plot((X[0], X[3]),(y_predicted[0], y_predicted[3]), 'r')
plt.show()
plt.close()
