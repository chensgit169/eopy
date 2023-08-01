import numpy as np
import mayavi.mlab as mlab

# 生成数据
x, y, z = np.mgrid[-5:5:50j, -5:5:50j, -5:5:50j]
s = np.sin(np.sqrt(x**2 + y**2 + z**2))

# 绘制等值面
mlab.figure(bgcolor=(1, 1, 1))  # 设置背景颜色为白色
mlab.contour3d(s, contours=10, colormap='viridis')
mlab.show()
