import matplotlib.pyplot as plt
import numpy as np

errors = np.sin(np.linspace(0,10,100))

outputs = errors * 0.5

plt.plot(errors, label="Lane Error")
plt.plot(outputs, label="Steering Output")

plt.legend()

plt.title("PID Steering Simulation")

plt.show()
