import os
from pathlib import Path
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window
from shapely.geometry import box

# --- Configuration ---
ORTHO_PATH = "C:/Users/sebastiaan_verbessel/Documents/test_senseforest/Ortho_BTR.tif"
VECTOR_PATH = "C:/Users/sebastiaan_verbessel/Documents/test_senseforest/ref_poly_BTR.gpkg"
OUTPUT_DIR = "C:/Users/sebastiaan_verbessel/Documents/test_senseforest/my_dataset/tiles_1024"
PATCH_DIR = os.path.join(OUTPUT_DIR, "patches")  # Single directory for both tiles and masks
METADATA_PATH = os.path.join(OUTPUT_DIR, "dataset_metadata_1024.csv")

PATCH_SIZE = 1024
STRIDE = 512
TARGET_CRS = "EPSG:31370"
biome_group = "Temperate"

os.makedirs(PATCH_DIR, exist_ok=True)

# 1. Load vector data and reproject directly to local target CRS
gdf = gpd.read_file(VECTOR_PATH)
if gdf.crs != TARGET_CRS:
    gdf = gdf.to_crs(TARGET_CRS)

gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid].copy()
sindex = gdf.sindex

metadata_rows = []

with rasterio.open(ORTHO_PATH) as src:
    # 2. Determine native dimensions & resolution reprojected into TARGET_CRS
    target_transform, target_width, target_height = calculate_default_transform(
        src.crs, TARGET_CRS, src.width, src.height, *src.bounds
    )

    # Pixel dimensions in meters (EPSG:31370 uses meters)
    res_x = abs(target_transform.a)
    res_y = abs(target_transform.e)
    avg_resolution_m = (res_x + res_y) / 2.0
    avg_resolution_cm = avg_resolution_m * 100.0

    # 3. Bin resolution: [1, 2) -> 1, [2, 3) -> 2, ..., [27, 28) -> 27
    resolution_bin = int(np.floor(avg_resolution_cm))
    if not (1 <= resolution_bin <= 27):
        print(f"Warning: Average resolution ({avg_resolution_cm:.2f} cm) falls outside the 1–28 cm range.")

    # 4. Virtual reprojection without fixed resampling size
    vrt_options = {
        "crs": TARGET_CRS,
        "transform": target_transform,
        "width": target_width,
        "height": target_height,
        "resampling": Resampling.bilinear,
    }

    with WarpedVRT(src, **vrt_options) as vrt:
        h, w = vrt.height, vrt.width

        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                win_transform = rasterio.windows.transform(window, vrt.transform)

                # Read RGB bands
                img = vrt.read([1, 2, 3], window=window)

                # Skip empty or mostly nodata tiles
                if np.all(img == 0) or np.mean(img == 0) > 0.6:
                    continue

                # Query intersecting polygons using spatial index
                win_bounds = rasterio.windows.bounds(window, vrt.transform)
                patch_box = box(*win_bounds)
                possible_matches_idx = list(sindex.intersection(patch_box.bounds))
                intersecting_gdf = gdf.iloc[possible_matches_idx]
                actual_matches = intersecting_gdf[intersecting_gdf.intersects(patch_box)]

                # Rasterize polygons (value 1)
                shapes = [(geom, 1) for geom in actual_matches.geometry]
                if shapes:
                    mask = rasterize(
                        shapes=shapes,
                        out_shape=(PATCH_SIZE, PATCH_SIZE),
                        transform=win_transform,
                        fill=0,
                        dtype=np.uint8,
                    )
                else:
                    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

                # Class balance statistics
                total_pixels = PATCH_SIZE * PATCH_SIZE
                deadwood_px = int(np.sum(mask == 1))
                frac_mortality = deadwood_px / total_pixels
                frac_background = 1.0 - frac_mortality
                mask_filled = 1 if frac_mortality > 0 else 0

                # Filenames and paths
                patch_id = f"patch_{x}_{y}_bin{resolution_bin}"
                tile_filename = f"{patch_id}.png"
                mask_filename = f"{patch_id}_mask.png"

                tile_rel_path = f"{tile_filename}"
                mask_rel_path = f"{mask_filename}"

                # Normalize / cast image format (RGB -> BGR for OpenCV)
                img_hwc = np.transpose(img, (1, 2, 0))
                if img_hwc.dtype != np.uint8:
                    max_val = img_hwc.max()
                    img_hwc = (img_hwc / (max_val if max_val > 0 else 1) * 255).astype(np.uint8)

                # Save tile and mask
                cv2.imwrite(
                    os.path.join(PATCH_DIR, tile_filename),
                    cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(os.path.join(PATCH_DIR, mask_filename), mask)

                # Direct inclusion of metadata attributes
                metadata_rows.append({
                    "id": "site_01",
                    "base_file_name": patch_id,
                    "global_file_path": PATCH_DIR,
                    "tile_path": tile_rel_path,
                    "mask_path": mask_rel_path,
                    "resolution_bin": resolution_bin,
                    "biome_group": biome_group,
                    "frac_mortality": frac_mortality,
                    "frac_background": frac_background,
                    "mask_filled": mask_filled,
                })

# Save metadata CSV directly
df_meta = pd.DataFrame(metadata_rows)
df_meta.to_csv(METADATA_PATH, index=False)

print(f"Generated {len(df_meta)} patches saved in '{PATCH_DIR}' (bin={resolution_bin}, biome={biome_group}).")
print("Saved metadata to:", METADATA_PATH)
if not df_meta.empty:
    print(df_meta[["base_file_name", "global_file_path", "tile_path", "frac_mortality"]].head())