import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the 2D Gaussian function
def gaussian_2d(x, y, t, x0, y0, sigma_x, sigma_y, amplitude, tau):
    return amplitude * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))) * np.exp(-t / tau)

# Create a grid of points
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Parameters for the Gaussian
x0 = 0
y0 = 0
sigma_x = 2
sigma_y = 2
amplitude = 1
tau = 0.5  # Fast time constant

# Create a figure and axes
fig, ax = plt.subplots()

# Initialize the plot
im = ax.imshow(gaussian_2d(X, Y, 0, x0, y0, sigma_x, sigma_y, amplitude, tau), cmap='hot', extent=[-10, 10, -10, 10])
ax.set_title('2D Gaussian Time Series')

# Animation update function
def update(t):
    im.set_data(gaussian_2d(X, Y, t, x0, y0, sigma_x, sigma_y, amplitude, tau))
    return [im]

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 5, 0.1), interval=50, blit=True)

# Show the animation
plt.show()