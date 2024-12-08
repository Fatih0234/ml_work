import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
# Feel free to change the number of data points or modify the relationship between x and y!
np.random.seed(42)  # Set seed for reproducibility
x = np.random.randn(100, 1)  # 100 data points
y = 5 * x + np.random.randn(100, 1)  # Linear relation with noise

# Parameters (feel free to experiment with these!)
w, b = 1000, 0.0  # Initial weight and bias (try different starting values)
learning_rate = 0.01  # Learning rate (e.g., 0.001, 0.1 to see its impact)
batch_size = 10  # Mini-batch size for Mini-Batch Gradient Descent (try 1, 50, or 100!)

# Batch Gradient Descent Function
def batch_descent(x, y, w, b, learning_rate):
    """
    Updates parameters using the full dataset (Batch Gradient Descent).
    """
    yhat = w * x + b
    loss = np.mean((y - yhat)**2)
    dw = -2 * np.mean(x * (y - yhat))
    db = -2 * np.mean(y - yhat)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b, loss

# Stochastic Gradient Descent Function
def stochastic_descent(x, y, w, b, learning_rate):
    """
    Updates parameters using one random data point at a time (Stochastic Gradient Descent).
    """
    idx = np.random.randint(0, len(x))  # Random index
    xi, yi = x[idx:idx+1], y[idx:idx+1]
    yhat = w * xi + b
    loss = (yi - yhat)**2
    dw = -2 * xi * (yi - yhat)
    db = -2 * (yi - yhat)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b, loss.item()

# Mini-Batch Gradient Descent Function
def mini_batch_descent(x, y, w, b, learning_rate, batch_size):
    """
    Updates parameters using a small random subset of data (Mini-Batch Gradient Descent).
    """
    idx = np.random.choice(len(x), batch_size, replace=False)  # Random mini-batch
    xi, yi = x[idx], y[idx]
    yhat = w * xi + b
    loss = np.mean((yi - yhat)**2)
    dw = -2 * np.mean(xi * (yi - yhat))
    db = -2 * np.mean(yi - yhat)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b, loss

# Iteration and training
# Experiment by changing the number of epochs to see how quickly each method converges!
epochs = 100  
losses_bgd, losses_sgd, losses_mini = [], [], []

for epoch in range(epochs):
    # Batch Gradient Descent
    w, b, loss_bgd = batch_descent(x, y, w, b, learning_rate)
    losses_bgd.append(loss_bgd)
    
    # Stochastic Gradient Descent
    w, b, loss_sgd = stochastic_descent(x, y, w, b, learning_rate)
    losses_sgd.append(loss_sgd)
    
    # Mini-Batch Gradient Descent
    w, b, loss_mini = mini_batch_descent(x, y, w, b, learning_rate, batch_size)
    losses_mini.append(loss_mini)
    
    # Visualization of predictions every 50 epochs
    if epoch % 50 == 0:
        plt.scatter(x, y, label="Data")
        plt.plot(x, w * x + b, color="red", label="Prediction")
        plt.title(f"Epoch {epoch}")
        plt.legend()
        plt.show()

# Compare loss curves
plt.plot(losses_bgd, label="BGD Loss")
plt.plot(losses_sgd, label="SGD Loss")
plt.plot(losses_mini, label="Mini-Batch Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Comparison")
plt.show()

# Experimentation Guidelines (Interactive Challenges):
# ===============================================
# Use these challenges to deepen your understanding of gradient descent methods:
# 
# 1. **Learning Rate:** Try different `learning_rate` values, such as 0.001, 0.1, or even 1.0.
#    - How does increasing or decreasing it affect convergence speed?
#    - Does a very large learning rate cause the model to overshoot the minimum?
# 
# 2. **Number of Epochs:** Increase `epochs` to 500 or 1000 and observe:
#    - How quickly does each method (BGD, SGD, Mini-Batch) converge?
#    - Does increasing epochs lead to better or worse results for SGD? Why?
# 
# 3. **Mini-Batch Size:** Experiment with different `batch_size` values, such as 1, 32, 64, or 100:
#    - What happens when the batch size is very small (close to SGD)?
#    - What happens when the batch size is very large (close to BGD)?
# 
# 4. **Initial Parameters:** Change the initial values of `w` and `b`:
#    - What happens if you set `w = 0` and `b = 0`?
#    - Does the choice of initial values affect convergence for SGD more than BGD? Why?
# 
# 5. **Data Complexity:** Modify the dataset by changing the relationship between `x` and `y`:
#    - For example, use `y = 3 * x**2 + np.random.randn(100, 1)` to simulate non-linear data.
#    - Can gradient descent still find a good approximation for non-linear relationships?
# 
# 6. **Compare Loss Curves:** Analyze the loss curves:
#    - Which method converges the fastest?
#    - Which method has the smoothest curve, and why?
#    - Can you explain why SGD is noisier than the others?
#
# 7. **Advanced Task (Bonus):**
#    - Modify the code to include momentum in gradient descent. Momentum helps accelerate 
#      convergence for SGD. How does it impact convergence speed and stability?
