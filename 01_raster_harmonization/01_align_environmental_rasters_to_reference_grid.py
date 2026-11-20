from pathlib import Path
import csv
import json
import logging
import math
import sys

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = Path(r"D:\KRB\KRB-DATASET")
INPUT_DIR = ROOT / "Harmonized_30m"
OUTPUT_DIR = ROOT / "Harmonized_30m" / "DEM_Reference_Harmonized"
REFERENCE_RASTER = INPUT_DIR / "DEM_30m.tif"

PREDICTORS = {
    "Clay_0_5cm": ("Clay_0_5cm_30m.tif", "continuous"),
    "Distance_to_Main_River": ("Distance_to_Main_River_30m.tif", "continuous"),
    "Drainage_Density": ("Drainage_Density_30m.tif", "continuous"),
    "ET": ("ET_30m.tif", "continuous"),
    "Max1Day_Rainfall": ("Max1Day_Rainfall_30m.tif", "continuous"),
    "Mean_Rainfall": ("Mean_Rainfall_30m.tif", "continuous"),
    "Sand_0_5cm": ("Sand_0_5cm_30m.tif", "continuous"),
    "Silt_0_5cm": ("Silt_0_5cm_30m.tif", "continuous"),
    "Slope": ("Slope_30m.tif", "continuous"),
    "SMAP": ("SMAP_30m.tif", "continuous"),
    "SPI": ("SPI_30m.tif", "continuous"),
    "TWI": ("TWI_30m.tif", "continuous"),
}

RESAMPLING = {
    "continuous": Resampling.bilinear,
    "categorical": Resampling.nearest,
}
NODATA_OUT = -9999.0
OUTPUT_DTYPE = "float32"
LOG_FILE = OUTPUT_DIR / "harmonization.log"
CSV_REPORT = OUTPUT_DIR / "Harmonization_QC_Report.csv"
JSON_METADATA = OUTPUT_DIR / "Reference_Grid_Metadata.json"


def configure_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def valid_values(array, nodata):
    mask = np.isfinite(array)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        mask &= array != nodata
    return mask


def raster_statistics(path):
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype("float64")
        values = data.compressed()
        if values.size == 0:
            return 0, None, None, None, None
        return (
            int(values.size),
            float(np.min(values)),
            float(np.max(values)),
            float(np.mean(values)),
            float(np.std(values)),
        )


def main():
    configure_logging()

    if not REFERENCE_RASTER.exists():
        raise FileNotFoundError(f"DEM reference raster not found: {REFERENCE_RASTER}")

    with rasterio.open(REFERENCE_RASTER) as ref:
        if ref.crs is None:
            raise ValueError("DEM reference raster has no CRS.")
        if ref.width <= 0 or ref.height <= 0:
            raise ValueError("DEM reference raster has invalid dimensions.")

        reference = {
            "crs": ref.crs.to_string(),
            "transform": list(ref.transform),
            "width": ref.width,
            "height": ref.height,
            "resolution_x": float(ref.res[0]),
            "resolution_y": float(ref.res[1]),
            "bounds": list(ref.bounds),
            "nodata": ref.nodata,
            "dtype": ref.dtypes[0],
        }
        ref_profile = ref.profile.copy()
        ref_mask = ref.dataset_mask() > 0
        ref_array = ref.read(1, masked=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_METADATA.write_text(json.dumps(reference, indent=2), encoding="utf-8")

    rows = []
    for variable, (filename, data_type) in PREDICTORS.items():
        source_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / f"{variable}_DEMgrid_30m.tif"

        if not source_path.exists():
            rows.append({
                "variable": variable,
                "source": str(source_path),
                "output": str(output_path),
                "status": "MISSING_INPUT",
            })
            logging.error("Missing input: %s", source_path)
            continue

        with rasterio.open(source_path) as src:
            if src.crs is None:
                raise ValueError(f"{filename} has no CRS.")
            source_transform = src.transform
            source_profile = {
                "crs": src.crs.to_string(),
                "width": src.width,
                "height": src.height,
                "resolution_x": float(src.res[0]),
                "resolution_y": float(src.res[1]),
                "bounds": list(src.bounds),
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
            }
            source = src.read(1).astype(np.float32)
            source_nodata = src.nodata

        source_valid = np.isfinite(source)
        if source_nodata is not None:
            if isinstance(source_nodata, float) and math.isnan(source_nodata):
                source_valid &= np.isfinite(source)
            else:
                source_valid &= source != source_nodata
        source = np.where(source_valid, source, NODATA_OUT).astype(np.float32)

        destination = np.full(
            (reference["height"], reference["width"]),
            NODATA_OUT,
            dtype=np.float32,
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=source_transform,
            src_crs=source_profile["crs"],
            src_nodata=NODATA_OUT,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=NODATA_OUT,
            resampling=RESAMPLING[data_type],
            init_dest_nodata=True,
            num_threads=2,
        )

        destination[~ref_mask] = NODATA_OUT
        destination[~np.isfinite(destination)] = NODATA_OUT

        profile = ref_profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype=OUTPUT_DTYPE,
            nodata=NODATA_OUT,
            compress="DEFLATE",
            predictor=3,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(destination.astype(np.float32), 1)
            dst.set_band_description(1, variable)
            dst.update_tags(
                variable=variable,
                reference_grid=str(REFERENCE_RASTER),
                resampling=data_type,
                resampling_method=RESAMPLING[data_type].name,
                processing="DEM-grid reprojection and alignment",
            )

        with rasterio.open(output_path) as check:
            aligned = (
                check.crs == ref_profile["crs"]
                and check.transform == ref_profile["transform"]
                and check.width == reference["width"]
                and check.height == reference["height"]
                and check.nodata == NODATA_OUT
            )
            arr = check.read(1)
            valid = arr != NODATA_OUT
            valid &= np.isfinite(arr)
            valid_count = int(valid.sum())
            if valid_count:
                v = arr[valid].astype(np.float64)
                minimum, maximum = float(v.min()), float(v.max())
                mean, std = float(v.mean()), float(v.std())
            else:
                minimum = maximum = mean = std = None

        status = "PASS" if aligned and valid_count > 0 else "FAIL"
        rows.append({
            "variable": variable,
            "source": str(source_path),
            "output": str(output_path),
            "status": status,
            "source_crs": source_profile["crs"],
            "source_width": source_profile["width"],
            "source_height": source_profile["height"],
            "source_res_x": source_profile["resolution_x"],
            "source_res_y": source_profile["resolution_y"],
            "source_nodata": source_profile["nodata"],
            "target_crs": reference["crs"],
            "target_width": reference["width"],
            "target_height": reference["height"],
            "target_res_x": reference["resolution_x"],
            "target_res_y": reference["resolution_y"],
            "resampling": RESAMPLING[data_type].name,
            "valid_pixels": valid_count,
            "minimum": minimum,
            "maximum": maximum,
            "mean": mean,
            "std": std,
            "grid_alignment": aligned,
        })
        logging.info("%s: %s | valid pixels=%s", variable, status, valid_count)

    fields = sorted({key for row in rows for key in row})
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    failed = [r for r in rows if r.get("status") != "PASS"]
    logging.info("Predictors processed: %d", len(rows))
    logging.info("Passed: %d | Failed/missing: %d", len(rows) - len(failed), len(failed))
    logging.info("QC report: %s", CSV_REPORT)
    logging.info("Reference metadata: %s", JSON_METADATA)

    if failed:
        raise RuntimeError(
            f"Harmonization completed with {len(failed)} failed or missing input(s). "
            f"See {CSV_REPORT}"
        )


if __name__ == "__main__":
    main()
