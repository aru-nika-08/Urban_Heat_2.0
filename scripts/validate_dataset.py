
import pandas as pd

df = pd.read_csv("outputs/TN_2023_Summer.csv")

print("INDEX RANGE VALIDATION")

for col in ["NDVI", "NDBI", "NDWI"]:
    invalid = df[(df[col] < -1) | (df[col] > 1)]

    print(f"\n{col}")
    print("Invalid records:", len(invalid))
    print("Percentage:", round(len(invalid) / len(df) * 100, 4), "%")
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())

print("\nEXTREME TEMPERATURE RECORDS")

hot = df[df["LST_C"] > 60]

print("Records above 60°C:", len(hot))
print(hot.head(10).to_string(index=False))

print("\nVALID OBSERVATION COUNTS")
print(df["ValidObs"].value_counts().sort_index())
