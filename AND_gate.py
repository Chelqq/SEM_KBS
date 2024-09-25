import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 0, 0, 1]) #puntos para mapa

theta = np.random.rand(3)  # 2 pesos + bias

#fn de costo, log-verosimilitud negativa
def cost_function(X, y, theta):
    h = sigmoid(np.dot(X, theta))
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))


# col de bias
col_bias = np.column_stack([np.ones(len(X)), X])

#act func sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient
    return theta

# Hyperparams
learning_rate = 0.1
iterations = 1000

# Training
theta = gradient_descent(col_bias, y, theta, learning_rate, iterations)

# Plot
plt.figure(figsize=(8, 6))

# plot de los puntos
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', label='Datos')

# plot de la linea
x_plot = np.array([np.min(X[:,0]), np.max(X[:,0])])
y_plot = (-1/theta[2]) * (theta[1] * x_plot + theta[0])
plt.plot(x_plot, y_plot, label='clasifiacion')

plt.title('Compuerta AND 2b')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
