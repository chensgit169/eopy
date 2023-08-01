import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
y1 = np.sin(2 * x + np.pi/2)
y2 = np.sin(2 * x + np.pi/4)
plt.plot(x, y1, '-b', x, y2, '--or')  # --: dashed line, o: circle, r: red
plt.show()
