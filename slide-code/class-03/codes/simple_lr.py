# ----------------------------------------
# Simple Linear Regression (1 feature)
# Written by Ye Kyaw Thu, LU Lab., Myanmar
# ----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create simple dataset (1D input)
x = np.array([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18]).reshape(-1, 1)
y = np.array([0.67, 0.85, 1.05, 1.0, 1.40, 1.5, 1.3, 1.54])

# Create model
model = LinearRegression()

# Train model (learn phi0 and phi1)
model.fit(x, y)

# Get learned parameters
phi0 = model.intercept_
phi1 = model.coef_[0]

print(f"Intercept (phi0): {phi0}")
print(f"Slope (phi1): {phi1}")

# Predict
x_line = np.linspace(0, 1.5, 100).reshape(-1, 1)
y_pred = model.predict(x_line)

# Plot
plt.scatter(x, y)
plt.plot(x_line, y_pred)
plt.title("Simple Linear Regression")
plt.xlabel("x")
plt.ylabel("y")

# Save figure
plt.savefig("simple_linear_regression.png", dpi=300, bbox_inches='tight')
#plt.show()
