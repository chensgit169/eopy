import numpy as np
from numpy import ndarray, tanh
import matplotlib.pyplot as plt


def stage_function(x: ndarray, a: float, s: float):
    return tanh(a * (x+s)) + tanh(-a * (x-s))


xs = np.arange(-8, 8, 0.01)
s = 2
plt.figure(dpi=256)
for a in np.arange(0, 3, 0.5):
    plt.plot(xs, stage_function(xs, a, s=s), label=f'a={a}')
plt.title(f'$f(x)=tanh(a(x+s))+tanh(-a(x-s))$, s={s}')
plt.legend()
plt.savefig('./stage_function.png')
