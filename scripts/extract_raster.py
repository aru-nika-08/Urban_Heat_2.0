
import rasterio
import numpy as np
import pandas as pd

raster_path = "data/raw/TN_2023_PostMonsoon_150m.tif"

with rasterio.open(raster_path) as src:

    # Read all five raster bands
    data = src.read()

    # Get pixel coordinates
    rows, cols = np.indices((src.height, src.width))

    # Convert pixel positions into geographic coordinates
    lon, lat = rasterio.transform.xy(
        src.transform, rows, cols, offset="center"
    )

    # Flatten the raster bands into columns
    df = pd.DataFrame({
        "Longitude": np.array(lon).ravel(),
        "Latitude": np.array(lat).ravel(),
        "LST_C": data[0].ravel(),
        "NDVI": data[1].ravel(),
        "NDBI": data[2].ravel(),
        "NDWI": data[3].ravel(),
        "ValidObs": data[4].ravel()
    })

    # Remove pixels without valid LST values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["LST_C"])

    # Save dataset
    df.to_csv(
        "outputs/TN_2023_PostMonsoon.csv",
        index=False
    )

    print("Extraction completed!")
    print("Dataset shape:", df.shape)
    print(df.head())
