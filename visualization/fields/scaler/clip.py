import numpy as np
import matplotlib.pyplot as plt

# 生成数据（包含NaN值）
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 在数据中引入NaN值
Z[X > 3] = np.nan

# 将NaN值转换为掩码值
Z_masked = np.ma.masked_invalid(Z)

# 绘制裁剪后的三维表面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z_masked, cmap='viridis', vmin=-1, vmax=1)

# 添加颜色条
fig.colorbar(surf)

# 添加标题和标签
ax.set_title('Clipped 3D Surface with NaN Values')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
