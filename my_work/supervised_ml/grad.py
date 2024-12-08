# Gradient Descent for Linear Regression
# yhat = wx + b
# loss = (y - yhat)**2 / N
import numpy as np
import matplotlib.pyplot as plt
# Initialize some parameters
x = np.random.randn(10, 1)
y = 5*x + np.random.randn()

# let's define a loss function
def loss_f(y, yhat):
    return np.mean((y - yhat)**2)

# Parameters
w = 0.0
b = 0.0

# Make predictions
yhat = w*x + b

# Plot with the loss lines for w and b
# plt.scatter(x, y)
# plt.plot(x, yhat, color='red')
# plt.show()


# Hyperparameters
learning_rate = 0.01

# Create gradient descent function

def descent(x, y, w, b, learning_rate):
    
    
    yhat = w*x + b
    loss = loss_f(y, yhat)
    dw = -2*np.mean(x*(y - yhat))
    db = -2*np.mean(y - yhat)
    w = w - learning_rate*dw
    b = b - learning_rate*db
    return w, b, loss

# Iteratively make updates

# for epoch in range(1000):
#     w, b, loss = descent(x, y, w, b, learning_rate)
    
#     # plot
#     if epoch % 50 == 0:
#         print(f'Epoch: {epoch}, w: {w}, b: {b}, loss: {loss}')
#         plt.scatter(x, y)
#         plt.plot(x, w*x + b, color='red')
#         plt.show()
