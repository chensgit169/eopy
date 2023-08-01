import numpy as np


class SingleCoil:
    def __init__(self, radius, current):
        self.radius = radius
        self.current = current

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.radius}, {self.current})')

    def __str__(self):
        return f'a coil with R={self.radius}, I={self.current}'


if __name__ == '__main__':
    coil = SingleCoil(1.0, 1.1)
    print(str(coil))
    print(repr(coil))
