import matplotlib.pyplot as plt
import numpy as np


# 定义要绘制的函数
def func(x, y, z):
    return np.sin(np.sqrt(x ** 2 + y ** 2 + z ** 2))


# 生成数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
z = np.linspace(-5, 5, 10)  # 创建多个z值，表示要绘制的切片平面的高度
X, Y, Z = np.meshgrid(x, y, z)

# 创建三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制多个切片图
for i in range(Z.shape[2]):
    slice_plane = Z[:, :, i]  # 获取当前切片平面的高度
    ax.contourf(X[:, :, i], Y[:, :, i], slice_plane, cmap='viridis')

# 添加颜色条
cbar = plt.colorbar(ax.collections[0])

# 添加标题和标签
ax.set_title('Function Slices in 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
