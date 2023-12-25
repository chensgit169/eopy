from shapely.geometry import LineString
import matplotlib.pyplot as plt

"""
Demonstration of using shapely to find intersection point of lines.  
"""

# 定义第一条折线段的坐标点
line1_coords = [(0, 0), (3, 3), (2, 1)]

# 定义第二条折线段的坐标点
line2_coords = [(0, 3), (3, 0)]

# 创建LineString对象
line1 = LineString(line1_coords)
line2 = LineString(line2_coords)

# 求两条线段的交点
intersection_point = line1.intersection(line2)

# 检查是否有交点并输出结果
if intersection_point.is_empty:
    print("两条线段没有交点")
else:
    intersection_coords = intersection_point.coords[0]
    print("交点坐标：", intersection_coords)
    # 可视化两条线段和交点
    x1, y1 = zip(*line1_coords)
    x2, y2 = zip(*line2_coords)
    xi, yi = intersection_point.coords[0]

    plt.plot(x1, y1, label='Line 1', color='blue')
    plt.plot(x2, y2, label='Line 2', color='green')
    plt.scatter(xi, yi, color='red', label='Intersection Point')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()
