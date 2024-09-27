import numpy as np
import matplotlib.pyplot as plt

# Datos
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 0, 0, 1])  # Puntos de la compuerta AND

# Columna de bias
col_bias = np.column_stack([np.ones(len(X)), X])

# Función sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Función de costo (log-verosimilitud negativa)
def cost_function(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    return -np.mean(y * np.log(h + 1e-9) + (1 - y) * np.log(1 - h + 1e-9))

# Differential Evolution
def differential_evolution(cost_func, bounds, X, y, pop_size=20, F=0.5, CR=0.7, generations=1000):
    dim = len(bounds)
    # Inicialización de la población aleatoria dentro de los límites dados
    population = np.random.rand(pop_size, dim)
    for i in range(dim):
        population[:, i] = bounds[i][0] + population[:, i] * (bounds[i][1] - bounds[i][0])
    
    best_solution = population[0]
    best_cost = cost_func(best_solution, X, y)
    
    for gen in range(generations):
        new_population = np.copy(population)
        for i in range(pop_size):
            # Selección de tres individuos distintos aleatorios (diferentes al actual)
            indices = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            
            # Mutación: vector de diferencia entre dos individuos
            mutant = a + F * (b - c)
            
            # Cruce: con probabilidad CR, seleccionar el gen mutante, sino mantener el original
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):  # Al menos un punto debe cruzarse
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])
            
            # Restringir el vector a los límites
            trial = np.clip(trial, [b[0] for b in bounds], [b[1] for b in bounds])
            
            # Evaluar el nuevo individuo (solución candidata)
            trial_cost = cost_func(trial, X, y)
            
            # Selección: si la nueva solución es mejor, la reemplaza
            if trial_cost < cost_func(population[i], X, y):
                new_population[i] = trial
                if trial_cost < best_cost:
                    best_solution = trial
                    best_cost = trial_cost
        
        # Actualizar la población
        population = new_population
    
    return best_solution

# Definimos los límites de los parámetros
bounds = [(-1, 1), (-1, 1), (-1, 1)]

# Aplicamos evolución diferencial
theta = differential_evolution(cost_function, bounds, col_bias, y, generations=1000)

# Visualización
plt.figure(figsize=(8, 6))

# Plot de los puntos
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', label='Datos')

# Plot de la línea de decisión
x_plot = np.array([np.min(X[:,0]), np.max(X[:,0])])
y_plot = (-1/theta[2]) * (theta[1] * x_plot + theta[0])
plt.plot(x_plot, y_plot, label='Clasificación')

plt.title('Compuerta AND 2b (Regresión Logística con Differential Evolution)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
