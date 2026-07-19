# CISP op vocabulary

Every op the harness accepts, with each field and its default. This
file is generated from `harnesscad.core.cisp.ops._REGISTRY` -- it is the
parser's own list, so an op or field absent here is absent from the
harness. Regenerate it whenever the op vocabulary changes.

A field shown with a default may be omitted from the JSON. Tuple
defaults such as `edges=()` are written as JSON arrays.

## Sketch setup

- `new_sketch` -- plane='XY'

## Sketch entities

- `add_arc` -- sketch='', cx=0.0, cy=0.0, r=1.0, start=0.0, end=90.0
- `add_circle` -- sketch='', cx=0.0, cy=0.0, r=1.0
- `add_ellipse` -- sketch='', cx=0.0, cy=0.0, rx=1.0, ry=0.5, rotation=0.0
- `add_line` -- sketch='', x1=0.0, y1=0.0, x2=0.0, y2=0.0
- `add_point` -- sketch='', x=0.0, y=0.0
- `add_polygon` -- sketch='', points=()
- `add_rectangle` -- sketch='', x=0.0, y=0.0, w=1.0, h=1.0
- `add_spline` -- sketch='', points=(), closed=False

## Constraints

- `constrain` -- kind='coincident', a='', b=None, value=None

## Solid creation

- `primitive` -- shape='box', dx=1.0, dy=1.0, dz=1.0, r=1.0, r2=0.0, h=1.0
- `extrude` -- sketch='', distance=1.0
- `revolve` -- sketch='', axis=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0), angle=360.0
- `loft` -- sketches=(), ruled=False, offsets=()
- `sweep` -- sketch='', path=''

## Combination

- `boolean` -- kind='union', target='', tool=''
- `hull` -- target='', tool=''
- `minkowski` -- radius=1.0
- `split` -- plane='XY', offset=0.0, keep='positive'

## Finishing

- `fillet` -- edges=(), radius=1.0
- `chamfer` -- edges=(), distance=1.0, distance2=None
- `shell` -- faces=(), thickness=1.0, kind='arc'
- `draft` -- faces=(), angle=0.0, neutral_plane=''
- `thicken` -- faces=(), thickness=1.0, both=False
- `hole` -- face_or_sketch='', x=0.0, y=0.0, diameter=1.0, depth=None, through=True, kind='simple', cbore_diameter=None, cbore_depth=None, csk_diameter=None, csk_angle=82.0

## Placement and repetition

- `transform` -- feature_or_body='', tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0
- `scale` -- feature_or_body='', sx=1.0, sy=1.0, sz=1.0
- `mirror` -- feature_or_body='', plane='XZ'
- `linear_pattern` -- feature='', direction=(1.0, 0.0, 0.0), count=2, spacing=1.0
- `circular_pattern` -- feature='', axis=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0), count=4, angle=360.0
- `pattern_transform` -- feature='', placements=()

## Assembly

- `add_instance` -- part='', x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0
- `mate` -- kind='rigid', a='', b='', value=None, base_port='', incoming_port='', base_port_type='', incoming_port_type=''

## Parameters

- `set_param` -- target=0, param='', value=None

Total: 34 ops.
