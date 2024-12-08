import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic data
np.random.seed(42)
x = np.random.randn(100, 1)  # 100 data points
y = 5 * x + np.random.randn(100, 1)  # Linear relation with noise

# Parameters
w, b = 1000, 0.0  # Initialize weight and bias
learning_rate = 0.01
batch_size = 10  # Mini-batch size for Mini-Batch GD

# Batch Gradient Descent Function
def batch_descent(x, y, w, b, learning_rate):
    yhat = w * x + b
    loss = np.mean((y - yhat)**2)
    dw = -2 * np.mean(x * (y - yhat))
    db = -2 * np.mean(y - yhat)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b, loss  # loss is already scalar here

# Stochastic Gradient Descent Function
def stochastic_descent(x, y, w, b, learning_rate):
    idx = np.random.randint(0, len(x))  # Random index
    xi, yi = x[idx:idx+1], y[idx:idx+1]
    yhat = w * xi + b
    loss = (yi - yhat)**2
    dw = -2 * xi * (yi - yhat)
    db = -2 * (yi - yhat)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b, loss.item()  # Extract scalar value

# Mini-Batch Gradient Descent Function
def mini_batch_descent(x, y, w, b, learning_rate, batch_size):
    idx = np.random.choice(len(x), batch_size, replace=False)  # Random mini-batch
    xi, yi = x[idx], y[idx]
    yhat = w * xi + b
    loss = np.mean((yi - yhat)**2)
    dw = -2 * np.mean(xi * (yi - yhat))
    db = -2 * np.mean(yi - yhat)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b, loss  # loss is scalar here


# Iterating and plotting
epochs = 100
losses_bgd, losses_sgd, losses_mini = [], [], []

for epoch in range(epochs):
    # Batch GD
    w, b, loss_bgd = batch_descent(x, y, w, b, learning_rate)
    losses_bgd.append(loss_bgd)
    
    # Stochastic GD
    w, b, loss_sgd = stochastic_descent(x, y, w, b, learning_rate)
    losses_sgd.append(loss_sgd)
    
    # Mini-Batch GD
    w, b, loss_mini = mini_batch_descent(x, y, w, b, learning_rate, batch_size)
    losses_mini.append(loss_mini)
    
    # Plot results periodically
    if epoch % 50 == 0:
        plt.scatter(x, y, label="Data")
        plt.plot(x, w * x + b, color="red", label="Model")
        plt.title(f"Epoch {epoch}")
        plt.legend()
        plt.show()

# Loss Visualization
plt.plot(losses_bgd, label="BGD Loss")
plt.plot(losses_sgd, label="SGD Loss")
plt.plot(losses_mini, label="Mini-Batch Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Comparison")
plt.show()
