import numpy as np
import matplotlib.pyplot as plt


# mesh, surf
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
xx, yy = np.meshgrid(x, y)
zz = np.sqrt(xx**2 + yy**2)


def contourf():
    """Filled contour plot in 2D."""
    plt.contourf(xx, yy, zz, levels=40)  # levels: number of contour lines
    plt.colorbar()
    plt.show()


def pcolor():
    """Pseudo-color map, similar to contourf,
    but does not draw contour lines."""
    plt.pcolormesh(xx, yy, zz)
    plt.colorbar()
    plt.show()


def surface():
    """3D surface plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # optional_cmap: viridis, plasma, inferno, magma, cividis, rainbow..
    ax.plot_surface(xx, yy, zz, cmap='gray')
    ax.plot_wireframe(xx, yy, zz, color='black', linewidths=0.5)
    plt.show()

contourf()

