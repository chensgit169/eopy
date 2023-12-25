# coding=utf-8

import ezdxf
import matplotlib.pyplot as plt
import numpy as np

from ezdxf.entities.line import Line
from ezdxf.entities.text import Text

dxf_file_path = "oblens.dxf"


"""
Read the AutoCAD file of the lens, and extract the coordinates of the polylines. 
"""

# TODO: standardize the layer names 规范图层名称


def extract_polyline_coordinates(file_dir):
    # 读取文件，模型化
    doc = ezdxf.readfile(file_dir)
    msp = doc.modelspace()

    texts = []
    layers = {}

    for entity in msp:
        if isinstance(entity, Line):
            # 获取线段起始点
            start_point = entity.dxf.start
            end_point = entity.dxf.end
            start_point = (start_point[0], start_point[1])
            end_point = (end_point[0], end_point[1])
            line_points = [start_point, end_point]

            # 按图层划分
            layer_name = entity.dxf.layer
            if layer_name in layers:
                layers[layer_name].append(line_points)
            else:
                layers[layer_name] = []

        elif entity.dxftype() == 'TEXT':
            # 文本标注暂先略过
            assert isinstance(entity, Text)
            x = entity.dxf.insert[0]
            y = entity.dxf.insert[1]
            label = entity.dxf.text
            texts.append({'x': x, 'y': y, 'label': label})
        else:
            print('Unknown type of entity in .xdf file:', entity.dxftype())
    return layers, texts


def plot_lens():
    layer_name = '2粗实线'

    coordinates, labels = extract_polyline_coordinates(dxf_file_path)

    for line in coordinates[layer_name]:
        line = np.array(line)
        plt.plot(line[:, 0], line[:, 1], 'black')
    # for label in labels:
    #     plt.text(label['x'], label['y'], label['label'])
    p0 = (304.6896471299115, 53.18093659519172)
    p1 = (214.6896471299115, 53.18093659519172)
    # plt.plot(*p0, '*')
    # plt.plot(*p1, '*')
    plt.title(layer_name, font='simhei', fontsize=14)
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], '--k')
    plt.savefig('./figures/lens.png', dpi=200)


def plot_mesh():
    layer_names = ['横线编号', '竖线编号']

    coordinates, labels = extract_polyline_coordinates(dxf_file_path)

    for layer_name in layer_names:
        for line in coordinates[layer_name]:
            line = np.array(line)
            plt.plot(line[:, 0], line[:, 1], '-*k')
    plt.title('网格(横线编号+竖线编号)', font='simhei', fontsize=14)
    plt.plot(*(54.68964712991146, 85.03734305574424), '*')
    plt.savefig('./figures/mesh.png', dpi=200)


def plot_vline():
    layer_names = ['竖线编号']
    coordinates, labels = extract_polyline_coordinates(dxf_file_path)

    for layer_name in layer_names:
        for line in coordinates[layer_name]:
            line = np.array(line)
            plt.plot(line[:, 0], line[:, 1], '-*k')
    plt.title('网格(横线编号+竖线编号)', font='simhei', fontsize=14)
    plt.savefig('./figures/vline.png', dpi=200)


def h_lines():
    layer_name = '横线编号'

    coordinates, labels = extract_polyline_coordinates(dxf_file_path)
    lines_xyxy = coordinates[layer_name]
    lines = []

    # Step1: adjust data
    for i, line in enumerate(lines_xyxy):
        p1, p2 = line
        if p1[0] > p2[0]:  # compare x(z)
            lines_xyxy[i] = [p2, p1]

    # Step2: iterate to link sections
    while len(lines_xyxy) > 0:
        sec0 = lines_xyxy.pop(0)
        lines.append([sec0])
        while True:
            found = False
            for i, sec1 in enumerate(lines_xyxy):
                if np.allclose(sec1[0], sec0[1]):
                    sec0 = lines_xyxy.pop(i)
                    lines[-1].append(sec0)
                    found = True
                    break
            if not found:
                break
    y0s = [line[0][0][1] for line in lines]
    lines = [lines[i] for i in np.argsort(y0s)[::-1]]
    for line in lines:
        line = [p[0] for p in line] + [line[-1][1]]
        xs, ys = zip(*line)
    #     plt.plot(xs, ys)
    #
    # plt.title(layer_name, font='simhei', fontsize=14)
    # plt.savefig('./figures/h_lines.png', dpi=200)
    return lines


def v_lines(y_max_lim: float = 160.):
    layer_name = '竖线编号'

    coordinates, labels = extract_polyline_coordinates(dxf_file_path)
    sections = coordinates[layer_name]

    # Step0: adjust pair order by y coordinates
    for j, line in enumerate(sections):
        p1, p2 = line
        if p1[1] > p2[1]:  # compare y(r), always let p1.y <= p2.y
            sections[j] = [p2, p1]

    # Step1: extract starting section
    lines = [[sec] for i, sec in enumerate(sections)
             if sec[1][1] >= y_max_lim]
    for line in lines:
        sections.remove(line[0])

    for i, line in enumerate(lines):
        sec0 = line[0]
        while True:
            for j, sec1 in enumerate(sections):
                if np.allclose(sec1[1], sec0[0]):
                    sec0 = sections.pop(j)
                    lines[i].append(sec0)
                    break
            else:
                break
    assert len(sections) == 0

    x0s = [line[0][0][0] for line in lines]
    lines = [lines[i] for i in np.argsort(x0s)]
    return lines


def intersect():
    from itertools import product
    from shapely.geometry import LineString
    inter_points = []

    for vs, hs in product(v_lines(), h_lines()):

        vps = [vs[0][-1]] + [v[0] for v in vs]
        vx, vy = zip(*vps)
        hps = [h[0] for h in hs] + [hs[-1][-1]]
        hx, hy = zip(*hps)

        plt.plot(vx, vy)
        plt.plot(hx, hy)

        for v, h in product(vs, hs):
            line1 = LineString(v)
            line2 = LineString(h)
            intersection_point = line1.intersection(line2, grid_size=0.01)
            if not intersection_point.is_empty:
                intersection_coords = intersection_point.coords[0]
                print(line1)
                print(line2)
                print("交点坐标：", intersection_coords)
                inter_points.append(intersection_coords)
                break
    for p in inter_points:
        plt.plot(*p, '*', color='red')
    plt.show()


if __name__ == "__main__":
    intersect()
    # plot_mesh()
