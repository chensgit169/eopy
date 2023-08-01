import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
xx, yy = np.meshgrid(x, y)
zz = xx * np.exp(-(xx**2 + yy**2))


def contour_3d():
    """3D contour plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    contour_plot = ax.contour(xx, yy, zz,
                              levels=20, cmap='viridis')

    # 添加颜色条
    plt.colorbar(contour_plot)

    # 添加标题和标签
    ax.set_title('3D Contour Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()


def contour_2d():
    """Filled contour plot in 2D."""
    plt.contourf(xx, yy, zz, levels=20)  # levels: number of contour lines
    plt.colorbar()
    plt.tight_layout()
    plt.show()


contour_3d()