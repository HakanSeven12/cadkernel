# cadkernel

A geometry kernel for CAD work: 2D curve algebra with a B-rep solid layer
built on top of it.

Early. The structure and the invariants are settled; most of the operations
are not written yet.

## Layering

```
acis    lift a SatDocument into brep, lower it back        [feature = "acis"]
  │
brep    owned mutable topology, SSI, boolean, blends       [feature = "brep"]
  │
geom2d  curves, intersection, offset, containment          [feature = "geom2d"]
```

`geom2d` is the default feature and compiles alone.

The direction is the point. Splitting a face during a boolean means
projecting the intersection curve into that face's `(u, v)` space and running
a loop boolean there — a 2D problem. The quality of `geom2d` therefore caps
the quality of `brep`, which is why 2D comes first rather than as an
afterthought.

## Two invariants

**Local frames.** Every operation lifts its input into a translation-only
local frame before doing arithmetic and shifts results back on the way out.
Drawings live at survey coordinates; an origin at 1.2e6 spends seven of
`f64`'s ~15 significant digits before anything has happened. What follows
does not look like a precision bug — arc fits quietly degrade to polylines,
stepping along a curve overflows to infinity, point-in-polygon flips sign at
an edge and a fill bleeds. Patching each site as it appears does not
converge. Lifting at the boundary does. See `geom2d::frame`.

**Provenance.** Every B-rep node remembers the record it was lifted from and
whether it has been edited since. Lowering writes clean nodes back as their
original bytes, so attributes, parameter-space curves and surface types the
kernel does not model survive an edit that did not touch them. Only dirtied
and newly built nodes are re-emitted. Without this, one boolean rewrites
every body in the file and quietly drops whatever the kernel cannot express.
See `brep::Provenance`.

## Scope

In: 2D curve intersection, offset, containment, chaining, area and length;
B-rep topology, surface–surface intersection, booleans; exact ACIS lift and
lower.

Out, for now: general fillet, chamfer and shelling. Constant-radius blends
between analytic faces are tractable and planned; corner blends where three
or more meet, variable radius, and blends on spline faces are not, and
pretending otherwise would be a poor use of anyone's time.

Analytic surfaces come first throughout. Plane, cone, cylinder, sphere and
torus intersect each other in closed form, and that covers most of the solids
that turn up in real drawings. NURBS-against-NURBS intersection is a separate
and much larger problem; a body this kernel cannot edit exactly should say so
rather than approximate quietly.

## Dependencies

None yet, deliberately. The intent is to take algorithms that are already
solved rather than rewrite them, but each one gets added when the code that
needs it lands:

| Crate | For |
| --- | --- |
| `robust` | exact orientation predicates — classification is where naive kernels break |
| `rstar` | spatial index for intersection candidate pruning and snapping |
| `cavalier_contours` | bulge-aware polyline offset and boolean |
| `iTriangle` | triangulation for fills |
| `curvo` | NURBS evaluation, derivatives, knot insertion |

Two decisions are still open: the vector math dependency, since the codec
uses `nalgebra` and the renderer consuming this uses `glam`, so one boundary
will need conversions either way; and whether the ACIS bridge depends on the
codec crate directly or inverts so the codec stays unaware of the kernel.

## Licence

MPL-2.0.
