
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# ==========================================
# 1. SETTINGS
# ==========================================

DATA_FOLDER = Path("outputs")
GRAPH_FOLDER = DATA_FOLDER / "graphs"

FOLDERS = {
    "temperature": GRAPH_FOLDER / "temperature",
    "indices": GRAPH_FOLDER / "indices",
    "temporal": GRAPH_FOLDER / "temporal",
    "correlation": GRAPH_FOLDER / "correlation"
}

for folder in FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

VARIABLES = ["LST_C", "NDVI", "NDBI", "NDWI"]
SEASONS = ["Summer", "Monsoon", "Post-monsoon"]

CHUNK_SIZE = 200000
SAMPLE_SIZE = 10000

# ==========================================
# 2. FIND AND PROCESS CSV FILES
# ==========================================

files = list(DATA_FOLDER.glob("TN_*.csv"))

summary_records = []
all_samples = []

for file in files:

    match = re.fullmatch(
        r"TN_(\d{4})_(Summer|Monsoon|PostMonsoon)",
        file.stem,
        re.IGNORECASE
    )

    if not match:
        continue

    year = int(match.group(1))
    season = match.group(2)

    if season.lower() == "postmonsoon":
        season = "Post-monsoon"

    print(f"\nProcessing {file.name}")

    total_pixels = 0
    sums = {col: 0.0 for col in VARIABLES}
    valid_obs_sum = 0.0
    valid_obs_count = 0
    samples = []

    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):

        for col in VARIABLES + ["ValidObs"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(
                    chunk[col], errors="coerce"
                )

        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=VARIABLES)

        if chunk.empty:
            continue

        total_pixels += len(chunk)

        for col in VARIABLES:
            sums[col] += chunk[col].sum()

        if "ValidObs" in chunk.columns:
            valid = chunk["ValidObs"].dropna()
            valid_obs_sum += valid.sum()
            valid_obs_count += len(valid)

        # Take a sample from each chunk for visualisation
        samples.append(
            chunk.sample(
                n=min(1000, len(chunk)),
                random_state=42
            )
        )

    if total_pixels == 0:
        print("No valid pixels found.")
        continue

    record = {
        "Year": year,
        "Season": season,
        "Pixels": total_pixels
    }

    for col in VARIABLES:
        record[col] = sums[col] / total_pixels

    record["Mean_ValidObs"] = (
        valid_obs_sum / valid_obs_count
        if valid_obs_count else np.nan
    )

    summary_records.append(record)

    sample_df = pd.concat(samples, ignore_index=True)

    sample_df["Year"] = year
    sample_df["Season"] = season

    all_samples.append(sample_df)

    print("Valid pixels:", total_pixels)

# ==========================================
# 3. CREATE SUMMARY DATASET
# ==========================================

if not summary_records:
    raise ValueError(
        "No matching seasonal CSV files found in outputs/"
    )

summary = pd.DataFrame(summary_records)

summary["Season"] = pd.Categorical(
    summary["Season"],
    categories=SEASONS,
    ordered=True
)

summary = summary.sort_values(["Year", "Season"])

sample_data = pd.concat(all_samples, ignore_index=True)

summary.to_csv(
    DATA_FOLDER / "eda_summary.csv",
    index=False
)

print("\n========== EDA SUMMARY ==========")
print(summary.to_string(index=False))

# ==========================================
# 4. GROUPED BAR CHART
# YEAR-WISE SEASONAL TEMPERATURE
# ==========================================

plt.figure(figsize=(11, 6))

sns.barplot(
    data=summary,
    x="Season",
    y="LST_C",
    hue="Year"
)

plt.title("Year-wise Seasonal Average LST")
plt.xlabel("Season")
plt.ylabel("Average Land Surface Temperature (°C)")
plt.legend(title="Year")
plt.tight_layout()

plt.savefig(
    FOLDERS["temporal"] / "yearwise_seasonal_lst.png",
    dpi=300
)
plt.close()

# ==========================================
# 5. SEASONAL COMPARISON FOR EACH YEAR
# ==========================================

for year in sorted(summary["Year"].unique()):

    year_data = summary[summary["Year"] == year]

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=year_data,
        x="Season",
        y="LST_C",
        order=SEASONS
    )

    plt.title(f"Seasonal LST Comparison - {year}")
    plt.xlabel("Season")
    plt.ylabel("Average LST (°C)")
    plt.tight_layout()

    plt.savefig(
        FOLDERS["temporal"] / f"seasonal_lst_{year}.png",
        dpi=300
    )
    plt.close()

# ==========================================
# 6. YEAR-WISE TEMPERATURE CHANGE
# ==========================================

for season in SEASONS:

    season_data = summary[
        summary["Season"] == season
    ].sort_values("Year")

    if len(season_data) < 2:
        continue

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=season_data,
        x="Year",
        y="LST_C"
    )

    plt.title(f"{season}: Temperature Across Years")
    plt.xlabel("Year")
    plt.ylabel("Average LST (°C)")
    plt.tight_layout()

    plt.savefig(
        FOLDERS["temporal"] / f"{season.lower()}_year_comparison.png",
        dpi=300
    )
    plt.close()

# Calculate year-to-year differences
change = summary.pivot(
    index="Season",
    columns="Year",
    values="LST_C"
)

change.to_csv(
    DATA_FOLDER / "seasonal_temperature_comparison.csv"
)

# ==========================================
# 7. TEMPERATURE HISTOGRAMS
# ==========================================

for (year, season), group in sample_data.groupby(
    ["Year", "Season"], observed=True
):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=group,
        x="LST_C",
        bins=40,
        kde=True
    )

    plt.title(f"LST Distribution - {season} {year}")
    plt.xlabel("Land Surface Temperature (°C)")
    plt.ylabel("Sampled Pixel Count")
    plt.tight_layout()

    plt.savefig(
        FOLDERS["temperature"] /
        f"histogram_{year}_{season}.png",
        dpi=300
    )
    plt.close()

# ==========================================
# 8. SEASONAL TEMPERATURE BOXPLOT
# ==========================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=sample_data,
    x="Season",
    y="LST_C",
    hue="Year",
    order=SEASONS
)

plt.title("Seasonal LST Distribution Across Years")
plt.xlabel("Season")
plt.ylabel("LST (°C)")
plt.tight_layout()

plt.savefig(
    FOLDERS["temperature"] / "seasonal_temperature_boxplot.png",
    dpi=300
)
plt.close()

# ==========================================
# 9. INDEX DISTRIBUTION HISTOGRAMS
# ==========================================

for col in ["NDVI", "NDBI", "NDWI"]:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=sample_data,
        x=col,
        hue="Season",
        bins=40,
        element="step",
        stat="density",
        common_norm=False
    )

    plt.title(f"{col} Distribution Across Seasons")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.tight_layout()

    plt.savefig(
        FOLDERS["indices"] / f"{col.lower()}_distribution.png",
        dpi=300
    )
    plt.close()

# ==========================================
# 10. LST VS NDVI SCATTER PLOT
# ==========================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=sample_data,
    x="NDVI",
    y="LST_C",
    hue="Season",
    alpha=0.4,
    s=15
)

plt.title("Land Surface Temperature vs NDVI")
plt.xlabel("NDVI")
plt.ylabel("LST (°C)")
plt.tight_layout()

plt.savefig(
    FOLDERS["indices"] / "lst_vs_ndvi.png",
    dpi=300
)
plt.close()

# ==========================================
# 11. LST VS NDBI SCATTER PLOT
# ==========================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=sample_data,
    x="NDBI",
    y="LST_C",
    hue="Season",
    alpha=0.4,
    s=15
)

plt.title("Land Surface Temperature vs NDBI")
plt.xlabel("NDBI")
plt.ylabel("LST (°C)")
plt.tight_layout()

plt.savefig(
    FOLDERS["indices"] / "lst_vs_ndbi.png",
    dpi=300
)
plt.close()

# ==========================================
# 12. CORRELATION HEATMAP
# ==========================================

correlation = sample_data[
    VARIABLES + ["ValidObs"]
].corr()

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("Environmental Variable Correlation")
plt.tight_layout()

plt.savefig(
    FOLDERS["correlation"] / "correlation_heatmap.png",
    dpi=300
)
plt.close()

# ==========================================
# 13. VALID OBSERVATION ANALYSIS
# ==========================================

if "ValidObs" in sample_data.columns:

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=sample_data,
        x="Season",
        y="ValidObs",
        hue="Year",
        order=SEASONS
    )

    plt.title("Valid Satellite Observations Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("Valid Observation Count")
    plt.tight_layout()

    plt.savefig(
        FOLDERS["temperature"] / "valid_observations.png",
        dpi=300
    )
    plt.close()

# ==========================================
# 14. FINISHED
# ==========================================

print("\n================================")
print("EDA ANALYSIS COMPLETED!")
print("================================")
print("Summary:", DATA_FOLDER / "eda_summary.csv")
print("Temperature graphs:", FOLDERS["temperature"])
print("Index graphs:", FOLDERS["indices"])
print("Temporal graphs:", FOLDERS["temporal"])
print("Correlation graphs:", FOLDERS["correlation"])
