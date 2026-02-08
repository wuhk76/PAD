# Python 3D Geometry Toolkit

A lightweight Python library for creating, transforming, visualizing, and exporting 3D geometry using simple, composable mesh operations. This project is designed for procedural geometry, educational use, and rapid prototyping without heavy 3D frameworks.

## Features

- **Geometric transformations**
  - Translation, rotation (Euler angles), scaling
  - Rotation around an arbitrary axis
  - Functional point-wise transforms
- **Mesh generation**
  - Primitive shapes (cube, grid)
  - Extrusion of edges, faces, or points
  - Revolving geometry around an axis
- **Mesh utilities**
  - Point simplification and clustering
  - Automatic edge and face joining by proximity
- **Visualization**
  - Quick 2D projection plotting using Matplotlib
- **File IO**
  - Import geometry from `.obj` files
  - Export points, lines, and faces to `.obj`

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install numpy matplotlib
```

## Geometry Representation

Geometry is represented as a list of elements, where each element is a list of 3D points:

```python
[
    [(x1, y1, z1), (x2, y2, z2), ...],
    ...
]
```

- 1 point → vertex
- 2 points → edge
- 3+ points → face

## Basic Usage

### Create and Transform Geometry

```python
from pad import cube, rotate, translate, scale

obj = cube()
obj = scale(obj, 2, 2, 2)
obj = rotate(obj, 0, 45, 0)
obj = translate(obj, 0, 0, 3)
```

### Extrude Geometry

```python
from pad import extrude

obj = extrude(obj, ((0, 0, 0), (0, 0, 1)), 2)
```

### Revolve Geometry Around an Axis

```python
from pad import revolve

obj = revolve(obj, ((0, 0, 0), (0, 1, 0)), 360, 36)
```

### Apply a Custom Function

```python
from pad import apply
import math

def wave(x, y, z):
    return (x, y, z + 0.2 * math.sin(x))

obj = apply(obj, wave)
```

## Visualization

```python
from pad import plot, show

plot(obj)
show()
```

## OBJ Import  Export

```python
from pad import openobj, exportobj

obj = openobj("model.obj")
exportobj(obj, "output.obj")
```

## Mesh Cleanup

```python
from pad import simplify, join

obj = simplify(obj, 0.01)
obj = join(obj, 0.05)
```

## License

MIT License
