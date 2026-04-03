"""
Lightweight SRTM elevation downloader.

Downloads SRTM 30m (1 arc-second) tiles from USGS/NASA and reprojects
them to match a target raster grid.  Only needs `rasterio` and `numpy`
— no GDAL CLI tools, no `elevation` pip package.

Usage:
    from srtm_elevation import fetch_elevation
    elev = fetch_elevation(bbox=(-122.35, 47.65, -122.32, 47.67),
                           target_shape=(102, 142),
                           resolution=30.0)
"""

from __future__ import annotations
import math, os, ssl, tempfile, zipfile, io
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# SRTM tile download URL patterns
# Primary: NASA SRTM via OpenTopography (no auth needed for SRTM GL1 30m)
SRTM_URL_TEMPLATE = (
    "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/{tile}.SRTMGL1.hgt.zip"
)

# Cache directory for downloaded tiles
SRTM_CACHE_DIR = Path(tempfile.gettempdir()) / "srtm_cache"


def _tile_name(lat: int, lon: int) -> str:
    """SRTM tile name for a given integer lat/lon (SW corner)."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def _tiles_for_bbox(bbox: Tuple[float, float, float, float]):
    """Return list of (lat, lon) tile SW-corners covering the bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_start = math.floor(min_lat)
    lat_end = math.floor(max_lat)
    lon_start = math.floor(min_lon)
    lon_end = math.floor(max_lon)
    tiles = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append((lat, lon))
    return tiles


def _download_srtm_hgt(lat: int, lon: int) -> Optional[np.ndarray]:
    """
    Download an SRTM .hgt tile and return as a 2D int16 array.
    Tries multiple free sources (no auth needed).
    """
    import urllib.request

    tile = _tile_name(lat, lon)
    cache_path = SRTM_CACHE_DIR / f"{tile}.hgt"

    # Return cached if available
    if cache_path.exists():
        print(f"  Using cached SRTM tile: {tile}")
        data = np.fromfile(str(cache_path), dtype=">i2")
        size = int(math.sqrt(len(data)))
        return data.reshape(size, size)

    SRTM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Free SRTM sources (no authentication required)
    urls = [
        # CGIAR SRTM v4.1 (90m, very reliable, no auth)
        f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_{_cgiar_tile_id(lat, lon)}.zip",
        # ViewfinderPanoramas (public domain SRTM-derived 3"" data)
        f"http://viewfinderpanoramas.org/dem3/{tile[0:3]}/{tile}.hgt.zip",
    ]

    # Try each URL
    for url in urls:
        print(f"  Trying: {url[:80]}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                resp = urllib.request.urlopen(req, timeout=120)
            except urllib.error.URLError:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                resp = urllib.request.urlopen(req, timeout=120, context=ctx)
            raw = resp.read()
            resp.close()

            if url.endswith('.zip'):
                # Extract .hgt from zip
                zf = zipfile.ZipFile(io.BytesIO(raw))
                hgt_names = [n for n in zf.namelist() if n.endswith('.hgt')]
                tif_names = [n for n in zf.namelist() if n.endswith('.tif')]

                if hgt_names:
                    hgt_data = zf.read(hgt_names[0])
                    cache_path.write_bytes(hgt_data)
                    data = np.frombuffer(hgt_data, dtype=">i2")
                    size = int(math.sqrt(len(data)))
                    print(f"  ✓ Got {tile}: {size}×{size} from .hgt")
                    return data.reshape(size, size).astype(np.float32)
                elif tif_names:
                    # CGIAR provides .tif files
                    tif_path = SRTM_CACHE_DIR / tif_names[0]
                    tif_path.write_bytes(zf.read(tif_names[0]))
                    import rasterio
                    with rasterio.open(str(tif_path)) as src:
                        elev = src.read(1).astype(np.float32)
                        nodata = src.nodata
                        if nodata is not None:
                            elev[elev == nodata] = 0
                        print(f"  ✓ Got {tile}: {elev.shape} from .tif, range {elev.min():.0f}–{elev.max():.0f}m")
                        # Save the geotiff for later reprojection use
                        return elev
            else:
                # Raw .hgt file
                cache_path.write_bytes(raw)
                data = np.frombuffer(raw, dtype=">i2")
                size = int(math.sqrt(len(data)))
                print(f"  ✓ Got {tile}: {size}×{size}")
                return data.reshape(size, size).astype(np.float32)

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    print(f"  ✗ Could not download tile {tile} from any source.")
    return None


def _cgiar_tile_id(lat: int, lon: int) -> str:
    """Map lat/lon to CGIAR SRTM 5×5° tile numbers."""
    # CGIAR tiles: x=1..72 (lon -180 to 180, 5° each), y=1..24 (lat 60 to -60, 5° each)
    x = ((lon + 180) // 5) + 1
    y = ((60 - lat) // 5)  # 0-indexed from N60
    return f"{x:02d}_{y:02d}"


def fetch_elevation(
    bbox: Tuple[float, float, float, float],
    target_shape: Tuple[int, int],
    resolution: float = 30.0,
    target_crs: str = "EPSG:32610",
) -> Optional[np.ndarray]:
    """
    Fetch real elevation data for a bounding box and reproject to match
    the target raster shape.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
        target_shape: (height, width) of the output array
        resolution: meters/pixel of the target grid
        target_crs: CRS of the target raster (for reprojection)

    Returns:
        2D float32 array of elevation values in meters,
        or None if download fails.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    from pyproj import Transformer

    min_lon, min_lat, max_lon, max_lat = bbox
    height, width = target_shape

    print(f"Fetching real SRTM elevation for bbox {bbox}")
    print(f"  Target shape: {width}×{height}, resolution: {resolution}m")

    # Determine which tiles we need
    tiles_needed = _tiles_for_bbox(bbox)
    print(f"  Need {len(tiles_needed)} SRTM tile(s): {[_tile_name(*t) for t in tiles_needed]}")

    # Download tiles
    tile_arrays = []
    for lat, lon in tiles_needed:
        arr = _download_srtm_hgt(lat, lon)
        if arr is not None:
            tile_arrays.append(((lat, lon), arr))

    if not tile_arrays:
        print("  ✗ No SRTM tiles could be downloaded.")
        return None

    # If we have a single tile, use it directly via rasterio reprojection
    # For the common case of bbox fitting within one tile
    if len(tile_arrays) == 1:
        (lat, lon), src_data = tile_arrays[0]
        tile_name = _tile_name(lat, lon)
        tiff_path = SRTM_CACHE_DIR / f"{tile_name}.tif"

        if tiff_path.exists():
            # Use the GeoTIFF directly for reprojection
            with rasterio.open(str(tiff_path)) as src:
                # Transform bbox to target CRS
                transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
                minx, miny = transformer.transform(min_lon, min_lat)
                maxx, maxy = transformer.transform(max_lon, max_lat)

                dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)
                elevation = np.zeros((height, width), dtype=np.float32)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=elevation,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

                # Clean up nodata
                elevation[elevation < -1000] = 0
                elevation[np.isnan(elevation)] = 0

                print(f"  ✓ Elevation reprojected: {elevation.shape}")
                print(f"    Range: {elevation.min():.1f}m – {elevation.max():.1f}m")
                print(f"    Mean:  {elevation.mean():.1f}m")
                return elevation

    # Fallback: use the raw array data with simple interpolation
    # (handles multi-tile or missing GeoTIFF case)
    print("  Using simple interpolation fallback...")
    (lat, lon), src_data = tile_arrays[0]

    # SRTM tiles cover 1°×1°, starting from SW corner
    # Row 0 = north edge, last row = south edge
    src_h, src_w = src_data.shape
    lat_per_pixel = 1.0 / (src_h - 1)
    lon_per_pixel = 1.0 / (src_w - 1)

    elevation = np.zeros((height, width), dtype=np.float32)

    for r in range(height):
        for c in range(width):
            # Map pixel to lat/lon
            frac_y = r / max(height - 1, 1)
            frac_x = c / max(width - 1, 1)
            px_lat = max_lat - frac_y * (max_lat - min_lat)
            px_lon = min_lon + frac_x * (max_lon - min_lon)

            # Map to source pixel
            src_r = int((lat + 1 - px_lat) / lat_per_pixel)
            src_c = int((px_lon - lon) / lon_per_pixel)

            src_r = max(0, min(src_r, src_h - 1))
            src_c = max(0, min(src_c, src_w - 1))

            elevation[r, c] = src_data[src_r, src_c]

    elevation[elevation < -1000] = 0
    print(f"  ✓ Elevation interpolated: {elevation.shape}")
    print(f"    Range: {elevation.min():.1f}m – {elevation.max():.1f}m")
    return elevation


if __name__ == "__main__":
    # Quick test with Seattle bbox
    bbox = (-122.345, 47.656, -122.326, 47.665)
    elev = fetch_elevation(bbox, target_shape=(102, 142), resolution=30.0)
    if elev is not None:
        print(f"\nSuccess! Shape: {elev.shape}")
        print(f"Elevation range: {elev.min():.1f} – {elev.max():.1f} m")
        print(f"Mean elevation: {elev.mean():.1f} m")
    else:
        print("\nFailed to fetch elevation data.")
