import pandas as pd

pd.set_option('display.max_rows', None)

# File paths to cleaned file data
file_economic = "../Fraser Institute Raw Data/cleaned_economic_data.csv"
file_freedom = "../Fraser Institute Raw Data/cleaned_human_freedom_idx.csv"
file_happiness = "../Happiness Index Raw data/cleaned_happiness_index.csv"
file_worldbank = "../World Development Indicators Data/cleaned_EUCountriesDevelopmentIndicators.csv"

# Load the data
df_economic = pd.read_csv(file_economic)
df_freedom  = pd.read_csv(file_freedom)
df_happiness = pd.read_csv(file_happiness)
df_worldbank = pd.read_csv(file_worldbank)

# Print head for user
print(df_economic.head())
print(df_freedom.head())
print(df_happiness.head())
print(df_worldbank.head())

# Print shapes for user
print(df_economic.shape)
print(df_freedom.shape)
print(df_happiness.shape)
print(df_worldbank.shape)


# Fraser Institute Economic Freedom Index data
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

# Remove the country_code column with is not needed
df_economic_cleaned = df_economic_cleaned.drop(columns=['country_code'])



# Fraser Institute Human Freedom Index data
# Get table information and remove any colums with missing data
print()
print(f"Cleaning data in {file_freedom}")

# Get missing counts for each column
missing_counts = df_freedom.isnull().sum()
print("Missing values per column:\n", missing_counts)

# Drop columns with missing data
df_freedom_cleaned = df_freedom.dropna(axis=1)
removed_columns = df_freedom.columns[df_freedom.isnull().any()]
print("Columns with missing data:", removed_columns)

# Remove the country_code column with is not needed
df_freedom_cleaned = df_freedom_cleaned.drop(columns=['country_code'])



# Happiness Index Report Data
# Get table information and remove any colums with missing data
print()
print(f"Cleaning data in {file_happiness}")

# Get missing counts for each column
missing_counts = df_happiness.isnull().sum()
print("Missing values per column:\n", missing_counts)

# Drop columns with missing data
df_happiness_cleaned = df_happiness.dropna(axis=1)
removed_columns = df_happiness.columns[df_happiness.isnull().any()]
print("Columns with missing data:", removed_columns)



# World Bank data
# Get table information and remove any colums with missing data
print()
print(f"Cleaning data in {file_worldbank}")

# Get missing counts for each column
missing_counts = df_worldbank.isnull().sum()
print("Missing values per column:\n", missing_counts)

# Drop columns with missing data
df_worldbank_cleaned = df_worldbank.dropna(axis=1)
removed_columns = df_worldbank.columns[df_worldbank.isnull().any()]
print("Columns with missing data:", removed_columns)


# Print new cleaned data for user
print(df_economic_cleaned.head())
print(df_freedom_cleaned.head())
print(df_happiness_cleaned.head())
print(df_worldbank_cleaned.head())


# Merge the data
merged_df = (
    df_economic_cleaned
    .merge(df_freedom_cleaned, on=["year", "country_name"], how="outer")
    .merge(df_happiness_cleaned, on=["year", "country_name"], how="outer")
    .merge(df_worldbank_cleaned, on=["year", "country_name"], how="outer")
)

# Drop any rows with missing data
merged_df_cleaned = merged_df.dropna()

# Print result for user
print()
print()
print()
print()
print(merged_df_cleaned.head())
print(merged_df_cleaned.shape)

# Save result to new csv file
output_file = "combined_dataset.csv"
merged_df_cleaned.to_csv(output_file, index=False)
