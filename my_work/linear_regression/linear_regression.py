import numpy as np
import matplotlib.pyplot as plt

# Generate random data
def generate_data():
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 1, size=x.shape)
    return x, y

# Plot the data
def plot_data(x, y):
    plt.scatter(x, y, label="Data Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Relationship")
    plt.legend()
    plt.show()

# Calculate L2 Loss (Mean Squared Error)
def l2_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Calculate L1 Loss (Mean Absolute Error)
def l1_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Gradient Descent for Linear Regression
def gradient_descent(x, y, learning_rate=0.01, epochs=100):
    w, b = 0.0, 0.0  # Initialize weights and bias
    losses = []

    for _ in range(epochs):
        # Predictions
        y_pred = w * x + b

        # Calculate gradients
        dw = -2 * np.mean(x * (y - y_pred))
        db = -2 * np.mean(y - y_pred)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Compute loss and store it
        losses.append(l2_loss(y, y_pred))

    return w, b, losses

# Plot loss reduction over epochs
def plot_loss_reduction(losses):
    plt.plot(range(len(losses)), losses, label="Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Gradient Descent Progress")
    plt.legend()
    plt.show()

# Plot fitted line
def plot_fitted_line(x, y, w, b):
    plt.scatter(x, y, label="Data Points")
    plt.plot(x, w * x + b, color='red', label="Fitted Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Fitted Line using Gradient Descent")
    plt.legend()
    plt.show()

# Main function
def main():
    # Generate data
    x, y = generate_data()

    # Plot the raw data
    plot_data(x, y)

    # Perform gradient descent
    learning_rate = 0.01
    epochs = 100
    w, b, losses = gradient_descent(x, y, learning_rate, epochs)

    # Plot the loss reduction over epochs
    plot_loss_reduction(losses)

    # Plot the fitted line
    plot_fitted_line(x, y, w, b)

    # Print final parameters
    print(f"Final weight (w): {w}")
    print(f"Final bias (b): {b}")
    
    # losses
    y_pred = w * x + b
    
    print(f"L2 Loss: {l2_loss(y, y_pred)}")
    print(f"L1 Loss: {l1_loss(y, y_pred)}")

    # Experimental Questions
    print("\nEXPERIMENTAL QUESTIONS:")
    print("1. What happens if you increase or decrease the learning rate? Try values like 0.001 or 0.1.")
    print("2. What happens if you add more noise to the data? Modify the random noise and observe.")
    print("3. How does the number of epochs impact the final loss? Try reducing it to 10 or increasing it to 500.")
    print("4. Can you modify the gradient descent function to use L1 loss instead of L2 loss?")

if __name__ == "__main__":
    main()
