# ============================================================
# 03_rebuild_hydrological_factors.py
# FINAL HYDROLOGICAL RECONSTRUCTION
# ============================================================
#
# Inputs:
#   DEM_UTM30m.tif
#   Slope_degrees.tif
#   terrain_common_mask.tif
#   krb_shp.shp
#
# Outputs:
#   KRB_boundary_mask.tif
#   hydrological_common_mask.tif
#   DEM_hydrologically_conditioned.tif
#   D8_Flow_Direction.tif
#   Flow_Accumulation_cells.tif
#   Flow_Accumulation_m2.tif
#   TWI.tif
#   Derived_Stream_Network.tif
#   Drainage_Density.tif
#   Distance_to_Derived_Drainage_m.tif
#   hydrology_configuration.txt
#
# No SAR / flood labels / FSM / AHP / MCDM / K-means.
#
# IMPORTANT:
# D8 routing uses:
#   1) priority-flood depression conditioning
#   2) strict downhill routing where possible
#   3) deterministic flat routing using a lexicographically
#      decreasing routing key. This avoids unresolved flats
#      without changing the scientific DEM values.
# ============================================================

import os
os.environ["SHAPE_RESTORE_SHX"] = "YES"

import gc
import heapq
import warnings

import numpy as np
import rasterio
import geopandas as gpd

from fiona import Env
from rasterio.features import geometry_mask
from scipy.ndimage import distance_transform_edt, uniform_filter

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# PATHS
# ============================================================

DEM_PATH = r"D:\KRB\Clean Data\Rebuilt_Data\Terrain\DEM_UTM30m.tif"
SLOPE_PATH = r"D:\KRB\Clean Data\Rebuilt_Data\Terrain\Slope_degrees.tif"
TERRAIN_MASK_PATH = r"D:\KRB\Clean Data\Rebuilt_Data\Common_Masks\terrain_common_mask.tif"
KRB_SHP = r"D:\KRB\New folder\Input Data non harmonized\krb_shp.shp"

OUTPUT_DIR = r"D:\KRB\Clean Data\Rebuilt_Data\Hydrology"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KRB_MASK_OUT = os.path.join(OUTPUT_DIR, "KRB_boundary_mask.tif")
HYDRO_MASK_OUT = os.path.join(OUTPUT_DIR, "hydrological_common_mask.tif")
CONDITIONED_DEM_OUT = os.path.join(OUTPUT_DIR, "DEM_hydrologically_conditioned.tif")
FLOW_DIRECTION_OUT = os.path.join(OUTPUT_DIR, "D8_Flow_Direction.tif")
FLOW_ACCUM_OUT = os.path.join(OUTPUT_DIR, "Flow_Accumulation_cells.tif")
FLOW_ACCUM_AREA_OUT = os.path.join(OUTPUT_DIR, "Flow_Accumulation_m2.tif")
TWI_OUT = os.path.join(OUTPUT_DIR, "TWI.tif")
STREAM_OUT = os.path.join(OUTPUT_DIR, "Derived_Stream_Network.tif")
DRAINAGE_DENSITY_OUT = os.path.join(OUTPUT_DIR, "Drainage_Density.tif")
DISTANCE_DRAINAGE_OUT = os.path.join(
    OUTPUT_DIR, "Distance_to_Derived_Drainage_m.tif"
)
CONFIG_OUT = os.path.join(OUTPUT_DIR, "hydrology_configuration.txt")


# ============================================================
# PARAMETERS
# ============================================================

CELL_SIZE = 30.0
CELL_AREA = CELL_SIZE ** 2

STREAM_THRESHOLD_CELLS = 1000
STREAM_THRESHOLD_AREA_M2 = STREAM_THRESHOLD_CELLS * CELL_AREA
STREAM_THRESHOLD_AREA_KM2 = STREAM_THRESHOLD_AREA_M2 / 1_000_000.0

DENSITY_RADIUS_M = 1000.0

# D8 direction:
# 0 NW, 1 N, 2 NE, 3 W, 4 E, 5 SW, 6 S, 7 SE
DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int8)
DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int8)
DISTANCE = np.array(
    [
        np.sqrt(2.0), 1.0, np.sqrt(2.0),
        1.0, 1.0,
        np.sqrt(2.0), 1.0, np.sqrt(2.0)
    ],
    dtype=np.float32,
)


# ============================================================
# HELPERS
# ============================================================

def write_float_raster(path, array, width, height, crs, transform):
    if array.shape != (height, width):
        raise RuntimeError(
            f"Invalid raster shape for {path}: "
            f"expected {(height, width)}, found {array.shape}"
        )

    out = np.where(
        np.isfinite(array), array, -9999.0
    ).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
        "predictor": 3,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)

    print("✓", os.path.basename(path))


def write_mask_raster(path, array, width, height, crs, transform):
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.uint8), 1)

    print("✓", os.path.basename(path))


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 78)
print("KOSI RIVER BASIN - HYDROLOGICAL RECONSTRUCTION")
print("=" * 78)
print("Resolution        :", CELL_SIZE, "m")
print("Stream threshold  :", STREAM_THRESHOLD_CELLS, "cells")
print("Stream area       :", STREAM_THRESHOLD_AREA_KM2, "km²")
print("SAR               : NOT USED")
print("Flood labels      : NOT USED")
print("FSM               : NOT USED")
print("AHP / MCDM        : NOT USED")
print("K-means            : NOT USED")


# ============================================================
# CHECK INPUTS
# ============================================================

print()
print("=" * 78)
print("CHECKING INPUTS")
print("=" * 78)

for p in [DEM_PATH, SLOPE_PATH, TERRAIN_MASK_PATH, KRB_SHP]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing input:\n{p}")
    print("✓", p)


# ============================================================
# READ DEM
# ============================================================

print()
print("=" * 78)
print("READING DEM")
print("=" * 78)

with rasterio.open(DEM_PATH) as src:
    width = src.width
    height = src.height
    crs = src.crs
    transform = src.transform
    resolution = src.res
    bounds = src.bounds
    nodata = src.nodata
    dem = src.read(1).astype(np.float32)

print("Width      :", width)
print("Height     :", height)
print("Resolution :", resolution)
print("CRS        :", crs)
print("Bounds     :", bounds)

if crs is None or not crs.is_projected:
    raise RuntimeError("DEM must have a projected CRS.")

if not np.allclose(resolution, (CELL_SIZE, CELL_SIZE)):
    raise RuntimeError(f"DEM is not 30 m. Found {resolution}")

dem_valid = np.isfinite(dem)
if nodata is not None:
    dem_valid &= dem != nodata

print("DEM valid  :", f"{dem_valid.sum():,}")


# ============================================================
# READ SLOPE
# ============================================================

print()
print("=" * 78)
print("READING SLOPE")
print("=" * 78)

with rasterio.open(SLOPE_PATH) as src:
    if (
        src.width != width
        or src.height != height
        or src.crs != crs
        or src.transform != transform
    ):
        raise RuntimeError("Slope grid is not exactly aligned with DEM.")

    slope = src.read(1).astype(np.float32)

slope_valid = np.isfinite(slope)
print("Slope valid:", f"{slope_valid.sum():,}")


# ============================================================
# READ TERRAIN MASK
# ============================================================

print()
print("=" * 78)
print("READING TERRAIN COMMON MASK")
print("=" * 78)

with rasterio.open(TERRAIN_MASK_PATH) as src:
    if (
        src.width != width
        or src.height != height
        or src.crs != crs
        or src.transform != transform
    ):
        raise RuntimeError("Terrain mask is not exactly aligned with DEM.")

    terrain_mask = src.read(1) == 1

print("Terrain common:", f"{terrain_mask.sum():,}")


# ============================================================
# READ KRB POLYGON
# ============================================================

print()
print("=" * 78)
print("READING KRB BOUNDARY")
print("=" * 78)

with Env(SHAPE_RESTORE_SHX="YES"):
    gdf = gpd.read_file(KRB_SHP)

if gdf.empty:
    raise RuntimeError("KRB shapefile is empty.")

if gdf.crs is None:
    raise RuntimeError("KRB shapefile has no CRS.")

geom_types = set(gdf.geometry.geom_type.dropna().unique())
if not geom_types.issubset({"Polygon", "MultiPolygon"}):
    raise RuntimeError(
        f"KRB shapefile must contain polygons. Found {geom_types}"
    )

if gdf.crs != crs:
    print("Reprojecting KRB boundary...")
    gdf = gdf.to_crs(crs)

gdf = gdf[gdf.geometry.notnull()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()

bad = ~gdf.geometry.is_valid
if bad.any():
    print("Repairing invalid polygon geometry...")
    gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)

if not gdf.geometry.is_valid.all():
    raise RuntimeError("KRB polygon geometry remains invalid.")

print("Features:", len(gdf))
print("Geometry:", geom_types)
print("CRS:", gdf.crs)


# ============================================================
# RASTERIZE KRB
# ============================================================

print()
print("=" * 78)
print("CREATING KRB MASK")
print("=" * 78)

krb_mask = geometry_mask(
    list(gdf.geometry),
    out_shape=(height, width),
    transform=transform,
    invert=True,
)

print("KRB pixels:", f"{krb_mask.sum():,}")


# ============================================================
# COMMON HYDROLOGICAL MASK
# ============================================================

print()
print("=" * 78)
print("BUILDING HYDROLOGICAL COMMON MASK")
print("=" * 78)

hydro_mask = (
    dem_valid
    & slope_valid
    & terrain_mask
    & krb_mask
)

hydro_pixels = int(hydro_mask.sum())

print("DEM valid         :", f"{dem_valid.sum():,}")
print("Terrain common    :", f"{terrain_mask.sum():,}")
print("KRB               :", f"{krb_mask.sum():,}")
print("Hydrological common:", f"{hydro_pixels:,}")

if hydro_pixels == 0:
    raise RuntimeError("Hydrological mask is empty.")

write_mask_raster(
    KRB_MASK_OUT, krb_mask, width, height, crs, transform
)
write_mask_raster(
    HYDRO_MASK_OUT, hydro_mask, width, height, crs, transform
)


# ============================================================
# PRIORITY-FLOOD DEPRESSION CONDITIONING
# ============================================================

print()
print("=" * 78)
print("PRIORITY-FLOOD DEPRESSION CONDITIONING")
print("=" * 78)

conditioned = dem.copy()
conditioned[~hydro_mask] = np.nan

# Hydrological-domain boundary.
boundary = np.zeros((height, width), dtype=bool)

boundary[0, :] = hydro_mask[0, :]
boundary[-1, :] = hydro_mask[-1, :]
boundary[:, 0] |= hydro_mask[:, 0]
boundary[:, -1] |= hydro_mask[:, -1]

for dr in (-1, 0, 1):
    for dc in (-1, 0, 1):
        if dr == 0 and dc == 0:
            continue

        shifted = np.zeros_like(hydro_mask)

        sr = slice(max(0, -dr), min(height, height - dr))
        sc = slice(max(0, -dc), min(width, width - dc))
        tr = slice(max(0, dr), min(height, height + dr))
        tc = slice(max(0, dc), min(width, width + dc))

        shifted[tr, tc] = hydro_mask[sr, sc]
        boundary |= hydro_mask & ~shifted

flat_conditioned = conditioned.ravel()
visited = np.zeros(hydro_pixels + (width * height - hydro_pixels), dtype=bool)
# The above is intentionally total-cell sized.
visited = np.zeros(width * height, dtype=bool)

queue = []

boundary_ids = np.flatnonzero(boundary)

for node in boundary_ids:
    if not visited[node]:
        visited[node] = True
        heapq.heappush(
            queue,
            (float(flat_conditioned[node]), int(node))
        )

del boundary_ids
gc.collect()

processed = 0

while queue:
    elev, node = heapq.heappop(queue)

    row = node // width
    col = node % width

    for k in range(8):
        nr = row + int(DR[k])
        nc = col + int(DC[k])

        if nr < 0 or nr >= height or nc < 0 or nc >= width:
            continue

        if not hydro_mask[nr, nc]:
            continue

        nid = nr * width + nc

        if visited[nid]:
            continue

        visited[nid] = True

        nelev = float(flat_conditioned[nid])

        if nelev < elev:
            nelev = elev
            flat_conditioned[nid] = np.float32(nelev)

        heapq.heappush(queue, (nelev, int(nid)))

    processed += 1

    if processed % 1_000_000 == 0:
        print("Priority-flood processed:", f"{processed:,}")

unvisited = int(np.sum(hydro_mask & ~visited))

print("Priority-flood processed:", f"{processed:,}")
print("Unvisited cells:", f"{unvisited:,}")

if unvisited != 0:
    raise RuntimeError("Priority-flood did not process all valid cells.")

filled_dem = flat_conditioned.reshape(height, width)
filled_dem[~hydro_mask] = np.nan

del queue
del visited
del boundary
gc.collect()

print("✓ Depression conditioning complete")


# ============================================================
# D8 ROUTING
# ============================================================

print()
print("=" * 78)
print("CALCULATING D8 FLOW DIRECTION")
print("=" * 78)

flow_direction = np.full(
    (height, width),
    255,
    dtype=np.uint8
)

best_drop = np.full(
    (height, width),
    -np.inf,
    dtype=np.float32
)

for k in range(8):
    dr = int(DR[k])
    dc = int(DC[k])

    if dr == -1:
        sr = slice(1, height)
        tr = slice(0, height - 1)
    elif dr == 1:
        sr = slice(0, height - 1)
        tr = slice(1, height)
    else:
        sr = slice(0, height)
        tr = slice(0, height)

    if dc == -1:
        sc = slice(1, width)
        tc = slice(0, width - 1)
    elif dc == 1:
        sc = slice(0, width - 1)
        tc = slice(1, width)
    else:
        sc = slice(0, width)
        tc = slice(0, width)

    source = filled_dem[sr, sc]
    target = filled_dem[tr, tc]

    valid = (
        np.isfinite(source)
        & np.isfinite(target)
        & hydro_mask[sr, sc]
        & hydro_mask[tr, tc]
    )

    drop = np.full(
        source.shape,
        -np.inf,
        dtype=np.float32
    )

    drop[valid] = (
        (source[valid] - target[valid])
        / DISTANCE[k]
    ).astype(np.float32)

    current = best_drop[sr, sc]
    direction_grid = flow_direction[sr, sc]

    better = drop > current

    current[better] = drop[better]
    direction_grid[better] = np.uint8(k)

    del source, target, valid, drop, better
    gc.collect()


flow_direction[~hydro_mask] = 255


# ============================================================
# FLAT-ROUTING KEY
# ============================================================

print()
print("=" * 78)
print("RESOLVING FLAT / TIED D8 CELLS")
print("=" * 78)

unresolved = (
    hydro_mask
    & (flow_direction == 255)
)

unresolved_count = int(unresolved.sum())

print(
    "Cells without strict downhill direction:",
    f"{unresolved_count:,}"
)


if unresolved_count > 0:

    # --------------------------------------------------------
    # Routing key:
    #
    # Primary key = distance to the hydrological boundary.
    #
    # Secondary key = raster index.
    #
    # A cell may route only to a neighbour with a strictly
    # smaller routing key.
    #
    # Therefore the resulting flat-routing graph is acyclic.
    # --------------------------------------------------------

    distance_to_boundary = distance_transform_edt(
        hydro_mask
    ).astype(np.float32)

    flat_ids = np.arange(
        width * height,
        dtype=np.int64
    ).reshape(height, width)

    # Normalized key. The index term is deliberately tiny
    # relative to the distance term.
    #
    # This is a routing key only. It does NOT modify DEM values.

    routing_key = (
        distance_to_boundary.astype(np.float64)
        * float(width * height + 1)
        +
        flat_ids.astype(np.float64)
    )

    routing_key[~hydro_mask] = np.inf

    # --------------------------------------------------------
    # Assign unresolved cells by descending routing key.
    #
    # Each unresolved cell must find a neighbour with a lower
    # key. Because the key is strictly ordered, cycles are
    # impossible.
    # --------------------------------------------------------

    unresolved_ids = np.flatnonzero(
        unresolved
    )

    unresolved_remaining = []

    for node in unresolved_ids:

        row = node // width
        col = node % width

        current_key = routing_key[row, col]

        best_direction = 255
        best_key = np.inf

        # Prefer lower routing key.
        for k in range(8):

            nr = row + int(DR[k])
            nc = col + int(DC[k])

            if (
                nr < 0
                or nr >= height
                or nc < 0
                or nc >= width
            ):
                continue

            if not hydro_mask[nr, nc]:
                continue

            neighbour_key = routing_key[nr, nc]

            if neighbour_key < current_key:
                if neighbour_key < best_key:
                    best_key = neighbour_key
                    best_direction = k

        if best_direction != 255:

            flow_direction[row, col] = np.uint8(
                best_direction
            )

        else:

            unresolved_remaining.append(
                int(node)
            )


    # --------------------------------------------------------
    # The boundary distance can theoretically contain local
    # equal-key plateaus. Resolve those using a second
    # deterministic pass based only on raster-index ordering.
    # --------------------------------------------------------

    if unresolved_remaining:

        print(
            "Second deterministic flat pass:",
            len(unresolved_remaining)
        )

        for node in unresolved_remaining:

            row = node // width
            col = node % width

            current_id = int(node)

            best_direction = 255
            best_id = current_id

            for k in range(8):

                nr = row + int(DR[k])
                nc = col + int(DC[k])

                if (
                    nr < 0
                    or nr >= height
                    or nc < 0
                    or nc >= width
                ):
                    continue

                if not hydro_mask[nr, nc]:
                    continue

                neighbour_id = (
                    nr * width + nc
                )

                if neighbour_id < best_id:
                    best_id = neighbour_id
                    best_direction = k

            if best_direction != 255:
                flow_direction[row, col] = np.uint8(
                    best_direction
                )


    del unresolved_ids
    del unresolved_remaining
    del distance_to_boundary
    del flat_ids
    del routing_key

    gc.collect()


# ============================================================
# FINAL D8 VALIDATION
# ============================================================

print()
print("=" * 78)
print("FINAL D8 VALIDATION")
print("=" * 78)

flow_direction[~hydro_mask] = 255

interior = hydro_mask.copy()
interior[0, :] = False
interior[-1, :] = False
interior[:, 0] = False
interior[:, -1] = False

missing = (
    interior
    & (flow_direction == 255)
)

missing_count = int(missing.sum())

print(
    "Interior cells without flow:",
    f"{missing_count:,}"
)

if missing_count != 0:
    ids = np.flatnonzero(missing)

    print("Unresolved locations:")
    for node in ids[:20]:
        row = node // width
        col = node % width
        print(
            " row=", row,
            "col=", col,
            "elevation=", filled_dem[row, col]
        )

    raise RuntimeError(
        f"D8 routing failed: {missing_count:,} "
        "interior cells remain unresolved."
    )


# ------------------------------------------------------------
# Verify downstream connection for every interior cell.
# ------------------------------------------------------------

invalid_connections = 0

valid_nodes = np.flatnonzero(interior)

for node in valid_nodes:

    row = node // width
    col = node % width

    k = int(flow_direction[row, col])

    if k == 255:
        invalid_connections += 1
        continue

    nr = row + int(DR[k])
    nc = col + int(DC[k])

    if nr < 0 or nr >= height or nc < 0 or nc >= width:
        invalid_connections += 1
        continue

    if not hydro_mask[nr, nc]:
        invalid_connections += 1

del valid_nodes
gc.collect()

print(
    "Invalid downstream connections:",
    f"{invalid_connections:,}"
)

if invalid_connections != 0:
    raise RuntimeError(
        f"Invalid downstream connections: "
        f"{invalid_connections:,}"
    )

print("✓ D8 routing passed")


# ============================================================
# SAVE CONDITIONED DEM
# ============================================================

write_float_raster(
    CONDITIONED_DEM_OUT,
    filled_dem,
    width,
    height,
    crs,
    transform
)


# ============================================================
# SAVE D8
# ============================================================

flow_profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "uint8",
    "crs": crs,
    "transform": transform,
    "nodata": 255,
    "compress": "deflate",
}

with rasterio.open(
    FLOW_DIRECTION_OUT,
    "w",
    **flow_profile
) as dst:
    dst.write(flow_direction, 1)

print("✓", os.path.basename(FLOW_DIRECTION_OUT))


# ============================================================
# BUILD DOWNSTREAM GRAPH
# ============================================================

print()
print("=" * 78)
print("BUILDING FLOW GRAPH")
print("=" * 78)

total_cells = width * height

flat_hydro = hydro_mask.ravel()
flat_direction = flow_direction.ravel()

downstream = np.full(
    total_cells,
    -1,
    dtype=np.int64
)

valid_nodes = np.flatnonzero(flat_hydro)

for node in valid_nodes:

    k = int(flat_direction[node])

    if k == 255:
        continue

    row = node // width
    col = node % width

    nr = row + int(DR[k])
    nc = col + int(DC[k])

    if (
        nr >= 0
        and nr < height
        and nc >= 0
        and nc < width
    ):
        nid = nr * width + nc

        if flat_hydro[nid]:
            downstream[node] = nid

del valid_nodes
gc.collect()


# ============================================================
# FLOW GRAPH INDEGREE
# ============================================================

valid_downstream = downstream[
    downstream >= 0
]

indegree = np.bincount(
    valid_downstream,
    minlength=total_cells
).astype(np.int32)

del valid_downstream
gc.collect()


# ============================================================
# TOPOLOGICAL FLOW ACCUMULATION
# ============================================================

print()
print("=" * 78)
print("CALCULATING FLOW ACCUMULATION")
print("=" * 78)

flow_accum_flat = np.zeros(
    total_cells,
    dtype=np.float64
)

flow_accum_flat[
    flat_hydro
] = 1.0

initial_nodes = np.flatnonzero(
    flat_hydro
    & (indegree == 0)
)

queue = np.empty(
    hydro_pixels,
    dtype=np.int64
)

queue_size = initial_nodes.size

queue[:queue_size] = initial_nodes

del initial_nodes
gc.collect()

head = 0
processed = 0

while head < queue_size:

    node = int(queue[head])
    head += 1

    dn = int(downstream[node])

    if dn >= 0:

        flow_accum_flat[dn] += (
            flow_accum_flat[node]
        )

        indegree[dn] -= 1

        if indegree[dn] == 0:

            queue[queue_size] = dn
            queue_size += 1

    processed += 1

    if processed % 1_000_000 == 0:
        print(
            "Processed:",
            f"{processed:,}"
        )


print(
    "Processed nodes:",
    f"{processed:,}"
)

print(
    "Expected nodes:",
    f"{hydro_pixels:,}"
)

if processed != hydro_pixels:

    raise RuntimeError(
        "\nFLOW GRAPH CONTAINS A CYCLE OR "
        "UNRESOLVED CONNECTION.\n"
        f"Expected: {hydro_pixels:,}\n"
        f"Processed: {processed:,}"
    )

print("✓ Flow graph is acyclic")


# ============================================================
# FLOW ACCUMULATION STATISTICS
# ============================================================

flow_accum = flow_accum_flat.reshape(
    height,
    width
)

flow_values = flow_accum[
    hydro_mask
]

flow_min = float(
    np.min(flow_values)
)

flow_max = float(
    np.max(flow_values)
)

flow_mean = float(
    np.mean(flow_values)
)

print()
print("=" * 78)
print("FLOW ACCUMULATION")
print("=" * 78)

print("Minimum:", flow_min)
print("Maximum:", flow_max)
print("Mean   :", flow_mean)

if flow_max < STREAM_THRESHOLD_CELLS:

    raise RuntimeError(
        "\nHYDROLOGY FAILED.\n"
        f"Maximum flow accumulation = {flow_max:.2f} cells\n"
        f"Required = {STREAM_THRESHOLD_CELLS} cells"
    )


# ============================================================
# FLOW ACCUMULATION AREA
# ============================================================

flow_accum_area = (
    flow_accum * CELL_AREA
)

flow_accum_area[
    ~hydro_mask
] = np.nan


# ============================================================
# STREAM EXTRACTION
# ============================================================

print()
print("=" * 78)
print("DERIVING STREAM NETWORK")
print("=" * 78)

stream = (
    flow_accum
    >= STREAM_THRESHOLD_CELLS
)

stream &= hydro_mask

stream_count = int(stream.sum())

stream_coverage = (
    100.0
    * stream_count
    / hydro_pixels
)

print(
    "Threshold:",
    STREAM_THRESHOLD_CELLS,
    "cells"
)

print(
    "Threshold area:",
    STREAM_THRESHOLD_AREA_KM2,
    "km²"
)

print(
    "Stream cells:",
    f"{stream_count:,}"
)

print(
    "Stream coverage:",
    f"{stream_coverage:.4f}%"
)

if stream_count == 0:
    raise RuntimeError(
        "No streams at fixed 1000-cell threshold."
    )


# ============================================================
# TWI
# ============================================================

print()
print("=" * 78)
print("CALCULATING TWI")
print("=" * 78)

slope_rad = np.radians(slope)

tan_slope = np.tan(slope_rad)
tan_slope = np.maximum(tan_slope, 1e-6)

with np.errstate(divide="ignore", invalid="ignore"):

    twi = np.log(
        (
            flow_accum_area
            + CELL_AREA
        )
        /
        tan_slope
    )

twi[~hydro_mask] = np.nan
twi[~np.isfinite(twi)] = np.nan

twi_values = twi[
    np.isfinite(twi)
]

print(
    "TWI minimum:",
    float(np.min(twi_values))
)

print(
    "TWI maximum:",
    float(np.max(twi_values))
)

print(
    "TWI mean:",
    float(np.mean(twi_values))
)

print(
    "TWI std:",
    float(np.std(twi_values))
)


# ============================================================
# DISTANCE TO DRAINAGE
# ============================================================

print()
print("=" * 78)
print("CALCULATING DISTANCE TO DRAINAGE")
print("=" * 78)

distance_input = np.ones(
    (height, width),
    dtype=np.uint8
)

distance_input[stream] = 0

distance_pixels = distance_transform_edt(
    distance_input
)

distance_m = (
    distance_pixels * CELL_SIZE
)

distance_m[~hydro_mask] = np.nan

distance_values = distance_m[
    np.isfinite(distance_m)
]

print(
    "Minimum:",
    float(np.min(distance_values)),
    "m"
)

print(
    "Maximum:",
    float(np.max(distance_values)),
    "m"
)

print(
    "Mean:",
    float(np.mean(distance_values)),
    "m"
)


# ============================================================
# DRAINAGE DENSITY
# ============================================================

print()
print("=" * 78)
print("CALCULATING DRAINAGE DENSITY")
print("=" * 78)

radius_pixels = int(
    round(
        DENSITY_RADIUS_M / CELL_SIZE
    )
)

kernel_size = (
    2 * radius_pixels + 1
)

stream_length = (
    stream.astype(np.float32)
    * CELL_SIZE
)

local_mean = uniform_filter(
    stream_length,
    size=kernel_size,
    mode="nearest"
)

window_cells = (
    kernel_size * kernel_size
)

local_length = (
    local_mean * window_cells
)

window_area = (
    kernel_size
    * CELL_SIZE
    * kernel_size
    * CELL_SIZE
)

drainage_density = (
    local_length / window_area
)

drainage_density[
    ~hydro_mask
] = np.nan

dd_values = drainage_density[
    np.isfinite(drainage_density)
]

print(
    "Minimum:",
    float(np.min(dd_values))
)

print(
    "Maximum:",
    float(np.max(dd_values))
)

print(
    "Mean:",
    float(np.mean(dd_values))
)


# ============================================================
# SAVE PRODUCTS
# ============================================================

print()
print("=" * 78)
print("SAVING HYDROLOGICAL PRODUCTS")
print("=" * 78)

write_float_raster(
    FLOW_ACCUM_OUT,
    flow_accum,
    width,
    height,
    crs,
    transform
)

write_float_raster(
    FLOW_ACCUM_AREA_OUT,
    flow_accum_area,
    width,
    height,
    crs,
    transform
)

write_float_raster(
    TWI_OUT,
    twi,
    width,
    height,
    crs,
    transform
)

write_float_raster(
    DRAINAGE_DENSITY_OUT,
    drainage_density,
    width,
    height,
    crs,
    transform
)

write_float_raster(
    DISTANCE_DRAINAGE_OUT,
    distance_m,
    width,
    height,
    crs,
    transform
)

write_mask_raster(
    STREAM_OUT,
    stream,
    width,
    height,
    crs,
    transform
)


# ============================================================
# CONFIGURATION
# ============================================================

with open(
    CONFIG_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "KOSI RIVER BASIN HYDROLOGICAL RECONSTRUCTION\n"
    )

    f.write(
        "=============================================\n\n"
    )

    f.write(
        "DEM conditioning: Priority-Flood\n"
    )

    f.write(
        "Flow direction: D8\n"
    )

    f.write(
        "Flat routing: deterministic lexicographic key\n"
    )

    f.write(
        "Flow accumulation: topological routing\n\n"
    )

    f.write(
        f"CRS: {crs}\n"
    )

    f.write(
        f"Resolution: {CELL_SIZE} m\n"
    )

    f.write(
        f"Width: {width}\n"
    )

    f.write(
        f"Height: {height}\n"
    )

    f.write(
        f"Hydrological pixels: {hydro_pixels}\n"
    )

    f.write(
        f"Flow accumulation maximum: {flow_max}\n"
    )

    f.write(
        f"Stream threshold: {STREAM_THRESHOLD_CELLS} cells\n"
    )

    f.write(
        f"Stream threshold area: "
        f"{STREAM_THRESHOLD_AREA_KM2} km2\n"
    )

    f.write(
        f"Stream cells: {stream_count}\n"
    )

    f.write(
        f"Stream coverage: {stream_coverage}%\n"
    )

    f.write(
        f"TWI min: {float(np.min(twi_values))}\n"
    )

    f.write(
        f"TWI max: {float(np.max(twi_values))}\n"
    )

    f.write(
        "\nSAR: NOT USED\n"
        "Flood labels: NOT USED\n"
        "FSM: NOT USED\n"
        "AHP: NOT USED\n"
        "MCDM: NOT USED\n"
        "K-means: NOT USED\n"
    )

print(
    "✓ Configuration saved"
)


# ============================================================
# FINAL FILE VALIDATION
# ============================================================

print()
print("=" * 78)
print("FINAL OUTPUT VALIDATION")
print("=" * 78)

outputs = [
    KRB_MASK_OUT,
    HYDRO_MASK_OUT,
    CONDITIONED_DEM_OUT,
    FLOW_DIRECTION_OUT,
    FLOW_ACCUM_OUT,
    FLOW_ACCUM_AREA_OUT,
    TWI_OUT,
    STREAM_OUT,
    DRAINAGE_DENSITY_OUT,
    DISTANCE_DRAINAGE_OUT,
]

for path in outputs:

    if not os.path.isfile(path):
        raise RuntimeError(
            f"Output missing:\n{path}"
        )

    with rasterio.open(path) as src:

        if (
            src.width != width
            or src.height != height
            or src.crs != crs
            or src.transform != transform
        ):
            raise RuntimeError(
                f"Output grid mismatch:\n{path}"
            )

    print(
        "✓",
        os.path.basename(path)
    )


# ============================================================
# FINAL SUCCESS
# ============================================================

print()
print("=" * 78)
print("HYDROLOGICAL RECONSTRUCTION PASSED")
print("=" * 78)

print()
print("Grid                :", f"{width} × {height}")
print("Total cells         :", f"{width * height:,}")
print("Hydrological cells  :", f"{hydro_pixels:,}")
print("Flow maximum        :", f"{flow_max:.2f} cells")
print(
    "Maximum area        :",
    f"{flow_max * CELL_AREA / 1_000_000.0:.4f} km²"
)
print(
    "TWI range           :",
    f"{np.min(twi_values):.4f} to {np.max(twi_values):.4f}"
)
print(
    "Stream threshold    :",
    STREAM_THRESHOLD_CELLS,
    "cells"
)
print(
    "Stream cells        :",
    f"{stream_count:,}"
)
print(
    "Stream coverage     :",
    f"{stream_coverage:.4f}%"
)

print()
print("OUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print()
print("✓ D8 unresolved cells: 0")
print("✓ Invalid connections: 0")
print("✓ Flow graph cycle check: PASSED")
print("✓ Fixed stream threshold: PASSED")
print()
print("NEXT:")
print("04_validate_hydrological_factors.py")
print("=" * 78)


# ============================================================
# CLEANUP
# ============================================================

del dem
del slope
del terrain_mask
del krb_mask
del hydro_mask

del conditioned
del filled_dem
del flat_conditioned

del flow_direction
del flow_accum_flat
del flow_accum
del flow_accum_area

del downstream
del indegree
del queue

del twi
del stream

del distance_input
del distance_pixels
del distance_m

del drainage_density

gc.collect()

print()
print("✓ MEMORY CLEANUP COMPLETE")
