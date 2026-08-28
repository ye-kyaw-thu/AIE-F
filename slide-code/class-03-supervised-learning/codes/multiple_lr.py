# ----------------------------------------
# Multiple Linear Regression with 3D Plot
# Written by Ye Kyaw Thu, LU Lab., Myanmar
# ----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Dataset (2 features)
X = np.array([
    [50, 1],
    [60, 2],
    [80, 2],
    [100, 3],
    [120, 3]
])

y = np.array([150, 180, 200, 250, 300])

# Train model
model = LinearRegression()
model.fit(X, y)

# Get parameters
phi0 = model.intercept_
phi = model.coef_

print(f"Intercept (phi0): {phi0}")
print(f"Weights (phi1, phi2): {phi}")

# Predict a new example
new_house = np.array([[90, 2]])
predicted_price = model.predict(new_house)

print(f"Predicted price: {predicted_price[0]}")

# Extract features
x1 = X[:, 0]
x2 = X[:, 1]

# Create mesh grid
x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1.min(), x1.max(), 10),
    np.linspace(x2.min(), x2.max(), 10)
)

# Predict plane
y_grid = model.predict(np.c_[x1_grid.ravel(), x2_grid.ravel()])
y_grid = y_grid.reshape(x1_grid.shape)

# Plot
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

# Scatter points
ax.scatter(x1, x2, y)

# Surface (plane)
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.3)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
#ax.set_zlabel('Output (y)')
ax.set_zlabel('Output (y)', labelpad=6)
ax.set_title("Multiple Linear Regression (3D Plane)")
#ax.view_init(elev=25, azim=135)

plt.tight_layout()
plt.savefig(
    "multiple_linear_regression_3D.png",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.3
)

#plt.show()

