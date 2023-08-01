import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

# 生成数据
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 三角化
tri = Triangulation(X.flatten(), Y.flatten())

# 绘制三维表面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(tri, Z.flatten(), cmap='viridis', edgecolor='k')
ax.axis('off')
# 添加标题和标签
ax.set_title('3D Triangular Surface Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
