import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
def rotate(object, ax, ay, az):
    ax = math.radians(ax)
    ay = math.radians(ay)
    az = math.radians(az)
    robject = []
    for j in range(len(object)):
        rpoints = []
        for point in object[j]:
            point = np.array(point)
            rotx = np.array(
                [[1, 0, 0],
                [0, math.cos(ax), -math.sin(ax)],
                [0, math.sin(ax), math.cos(ax)]])
            roty = np.array(
                [[math.cos(ay), 0, math.sin(ay)],
                [0, 1, 0],
                [-math.sin(ay), 0, math.cos(ay)]])
            rotz = np.array(
                [[math.cos(az), -math.sin(az), 0],
                [math.sin(az), math.cos(az), 0],
                [0, 0, 1]])
            rpoint = tuple(rotz @ roty @ rotx @ point)
            rpoints.append(rpoint)
        robject.append(rpoints)
    return robject
def translate(object, tx, ty, tz):
    tobject = []
    for j in range(len(object)):
        tpoints = []
        for point in object[j]:
            point = np.array(point)
            tpoint = tuple(point + np.array([tx, ty, tz]))
            tpoints.append(tpoint)
        tobject.append(tpoints)
    return tobject
def scale(object, sx, sy, sz):
    sobject = []
    for j in range(len(object)):
        spoints = []
        for point in object[j]:
            point = np.array(point)
            spoint = tuple(point * np.array([sx, sy, sz]))
            spoints.append(spoint)
        sobject.append(spoints)
    return sobject
def apply(object, f):
    transform = []
    for j in range(len(object)):
        points = []
        for point in object[j]:
            try:
                points.append(f(point[0], point[1], point[2]))
            except ZeroDivisionError:
                pass
        transform.append(points)
    return transform
def revaxis(point, p1, axis, angle):
    angle = math.radians(angle)
    point = np.array(point)
    p1 = np.array(p1)
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    v = point - p1
    cost = math.cos(angle)
    sint = math.sin(angle)
    vrot = (
        v * cost +
        np.cross(axis, v) * sint +
        axis * np.dot(axis, v) * (1 - cost)
    )
    return tuple(vrot + p1)
def revolve(object, l, angle, d):
    p1, p2 = l
    axis = np.array(p2) - np.array(p1)
    result = []
    step = angle / d
    for i in range(d + 1):
        a = step * i
        robject = []
        for face in object:
            rface = []
            for point in face:
                rface.append(
                    revaxis(point, p1, axis, a)
                )
            robject.append(rface)
        result.extend(robject)
    return result
def extrude(object, l, d):
    p1, p2 = l
    direction = np.array(p2) - np.array(p1)
    norm = np.linalg.norm(direction)
    direction = direction / norm
    offset = direction * d
    result = []
    for elem in object:
        n = len(elem)
        telem = [tuple(np.array(p) + offset) for p in elem]
        if n == 1:
            result.append([elem[0], telem[0]])
        elif n == 2:
            p1, p2 = elem
            q1, q2 = telem
            result.append([p1, p2, q2, q1])
        else:
            result.append(elem)
            result.append(telem)
            for i in range(n):
                a = elem[i]
                b = elem[(i + 1) % n]
                c = telem[(i + 1) % n]
                d = telem[i]
                result.append([a, b, c, d])
    return result
def simplify(object, d):
    d2 = d * d
    points = []
    for elem in object:
        for p in elem:
            points.append(np.array(p, dtype = float))
    n = len(points)
    used = [False] * n
    clusters = []
    for i in range(n):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, n):
            if used[j]:
                continue
            if np.sum((points[i] - points[j]) ** 2) <= d2:
                cluster.append(j)
                used[j] = True
        clusters.append(cluster)
    centroids = {}
    for cluster in clusters:
        avg = sum(points[i] for i in cluster) / len(cluster)
        avg = tuple(avg)
        for i in cluster:
            centroids[i] = avg
    result = []
    idx = 0
    for elem in object:
        nelem = []
        for j in range(len(elem)):
            nelem.append(centroids[idx])
            idx += 1
        result.append(nelem)
    return result
def join(object, d):
    d2 = d * d
    points = []
    for elem in object:
        for p in elem:
            if p not in points:
                points.append(p)
    pts = [np.array(p, dtype=float) for p in points]
    n = len(pts)
    result = []
    result.extend(object)
    edges = set()
    faces = set()
    for i in range(n):
        for j in range(i + 1, n):
            if np.sum((pts[i] - pts[j]) ** 2) <= d2:
                edge = tuple(sorted((points[i], points[j])))
                edges.add(edge)
    for i, j, k in itertools.combinations(range(n), 3):
        if (
            np.sum((pts[i] - pts[j]) ** 2) <= d2 and
            np.sum((pts[j] - pts[k]) ** 2) <= d2 and
            np.sum((pts[k] - pts[i]) ** 2) <= d2
        ):
            face = tuple(sorted((points[i], points[j], points[k])))
            faces.add(face)
    for e in edges:
        result.append(list(e))
    for f in faces:
        result.append(list(f))
    return result
def plot(object, type = '-', color = 'k', equal = True):
    form = object
    if equal:
        plt.axis('equal')
    for points in object:
        xvals = [point[0] for point in points] + [points[0][0]]
        yvals = [point[2] for point in points] + [points[0][2]]
        plt.plot(xvals, yvals, type, color = color)
def show():
    plt.show()
def pause(t):
    plt.pause(t)
def clf():
    plt.clf()
def openobj(filename):
    vertices = []
    obj = []
    with open(filename, 'r') as file:
        for line in file:
            if not line or line.startswith('#'):
                continue
            parts = line.strip().split()
            if parts[0] == 'v':
                vertices.append((
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3])
                ))
            elif parts[0] == 'f':
                face = []
                for vert in parts[1:]:
                    idx = int(vert.split('/')[0]) - 1
                    face.append(vertices[idx])
                obj.append(face)
    return obj
def exportobj(points, path):
    vertexmap = {}
    vertices = []
    elements = []
    def getindex(v):
        if v not in vertexmap:
            vertexmap[v] = len(vertices) + 1
            vertices.append(v)
        return vertexmap[v]
    for elem in points:
        if not elem:
            continue
        indices = [getindex(v) for v in elem]
        if len(indices) == 1:
            elements.append(('p', indices))
        elif len(indices) == 2:
            elements.append(('l', indices))
        else:
            elements.append(('f', indices))
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for kind, idxs in elements:
            f.write(kind + ' ' + ' '.join(map(str, idxs)) + '\n')
def cube():
    faces = [
    [
        (-1, -1, -1),
        (-1,  1, -1),
        (-1,  1,  1),
        (-1, -1,  1),
    ],
    [
        (1, -1, -1),
        (1,  1, -1),
        (1,  1,  1),
        (1, -1,  1),
    ],
    ]
    fig = faces + rotate(faces, 0, 90, 0) + rotate(faces, 0, 0, 90)
    return fig
def grid(x, y, d):
    faces = []
    dx = x / d
    dy = y / d
    x0 = -x / 2
    y0 = -y / 2
    for i in range(d):
        for j in range(d):
            x1 = x0 + i * dx
            x2 = x1 + dx
            y1 = y0 + j * dy
            y2 = y1 + dy
            face = [
                (x1, y1, 0),
                (x2, y1, 0),
                (x2, y2, 0),
                (x1, y2, 0),
            ]
            faces.append(face)
    return faces
def polar(n, f):
    avals = np.linspace(0, 2 * math.pi, n + 1)
    rvals = [f(a) for a in avals]
    points = []
    for j in range(len(avals)):
        points.append((rvals[j] * math.cos(avals[j]), rvals[j] * math.sin(avals[j]), 0))
    return [points]
