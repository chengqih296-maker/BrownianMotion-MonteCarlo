
# Monte Carlo Simulation of 2D Brownian Motion
# Author: Ryan He

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n_steps = 1000
n_particles = 100
dt = 1
D_real = 1.0

np.random.seed(42)

dx = np.sqrt(2 * D_real * dt) * np.random.randn(n_steps, n_particles)
dy = np.sqrt(2 * D_real * dt) * np.random.randn(n_steps, n_particles)

x = np.cumsum(dx, axis=0)
y = np.cumsum(dy, axis=0)

plt.figure(figsize=(8, 6))
for i in range(5):
    plt.plot(x[:, i], y[:, i], label=f'Particle {i+1}')
plt.title('2D Brownian Motion Trajectories')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid()
plt.show()

msd = np.mean(x**2 + y**2, axis=1)

time = np.arange(1, n_steps + 1) * dt
plt.figure(figsize=(8, 6))
plt.plot(time, msd, label='MSD')
plt.xlabel('Time')
plt.ylabel('Mean Squared Displacement')
plt.title('MSD vs Time')
plt.grid()
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(time, msd)
D_estimated = slope / 4

print(f"True D = {D_real}")
print(f"Estimated D = {D_estimated:.4f}")
