import pandas as pd

pd.set_option('display.max_rows', None)

# File paths
file_economic = "../Fraser Institute Raw Data/cleaned_economic_data.csv"
file_freedom = "../Fraser Institute Raw Data/cleaned_human_freedom_idx.csv"
file_happiness = "../Happiness Index Raw data/cleaned_happiness_index.csv"
file_worldbank = "../World Development Indicators Data/cleaned_EUCountriesDevelopmentIndicators.csv"

# Load the data
df_economic = pd.read_csv(file_economic)
df_freedom  = pd.read_csv(file_freedom)
df_happiness = pd.read_csv(file_happiness)
df_worldbank = pd.read_csv(file_worldbank)

print(df_economic.head())
print(df_freedom.head())
print(df_happiness.head())
print(df_worldbank.head())

print(df_economic.shape)
print(df_freedom.shape)
print(df_happiness.shape)
print(df_worldbank.shape)

# Get table information and remove any colums with missing data
print()
print(f"Cleaning data in {file_economic}")

# Get missing counts for each column
missing_counts = df_economic.isnull().sum()
print("Missing values per column:\n", missing_counts)

# Drop columns with missing data
df_economic_cleaned = df_economic.dropna(axis=1)
removed_columns = df_economic.columns[df_economic.isnull().any()]
print("Columns with missing data:", removed_columns)

df_economic_cleaned = df_economic_cleaned.drop(columns=['country_code'])


print()
print(f"Cleaning data in {file_freedom}")

missing_counts = df_freedom.isnull().sum()
print("Missing values per column:\n", missing_counts)

df_freedom_cleaned = df_freedom.dropna(axis=1)
removed_columns = df_freedom.columns[df_freedom.isnull().any()]
print("Columns with missing data:", removed_columns)

df_freedom_cleaned = df_freedom_cleaned.drop(columns=['country_code'])



print()
print(f"Cleaning data in {file_happiness}")

missing_counts = df_happiness.isnull().sum()
print("Missing values per column:\n", missing_counts)

df_happiness_cleaned = df_happiness.dropna(axis=1)
removed_columns = df_happiness.columns[df_happiness.isnull().any()]
print("Columns with missing data:", removed_columns)


print()
print(f"Cleaning data in {file_worldbank}")
missing_counts = df_worldbank.isnull().sum()
print("Missing values per column:\n", missing_counts)

df_worldbank_cleaned = df_worldbank.dropna(axis=1)
removed_columns = df_worldbank.columns[df_worldbank.isnull().any()]
print("Columns with missing data:", removed_columns)

print(df_economic_cleaned.head())
print(df_freedom_cleaned.head())
print(df_happiness_cleaned.head())
print(df_worldbank_cleaned.head())


# Drop unwanted or common fields

# combined = combined.drop(columns=['column_to_drop_y'])



# Merge the data
merged_df = (
    df_economic_cleaned
    .merge(df_freedom_cleaned, on=["year", "country_name"], how="outer")
    .merge(df_happiness_cleaned, on=["year", "country_name"], how="outer")
    .merge(df_worldbank_cleaned, on=["year", "country_name"], how="outer")
)


merged_df_cleaned = merged_df.dropna()

print()
print()
print()
print()
print(merged_df_cleaned.head())
print(merged_df_cleaned.shape)

output_file = "combined_dataset.csv"
merged_df_cleaned.to_csv(output_file, index=False)
