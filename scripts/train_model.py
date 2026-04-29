from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HEALTH_FILE = Path("data/raw/pa_county_health_2025.xlsx")
ACS_FILE = Path("data/raw/pa_income_poverty_acs.csv")

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("model")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df, possible_names):
    clean_columns = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        key = name.strip().lower()
        if key in clean_columns:
            return clean_columns[key]

    return None


def to_number(series):
    return pd.to_numeric(series, errors="coerce")


def minmax_score(series, higher_is_better=True):
    series = to_number(series)
    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        score = pd.Series(50, index=series.index)
    else:
        score = (series - min_value) / (max_value - min_value) * 100

    if not higher_is_better:
        score = 100 - score

    return score


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "model_name": name,
        "model": model,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


if not HEALTH_FILE.exists():
    raise FileNotFoundError(
        "Health Rankings dataset not found. Make sure it is saved as "
        "data/raw/pa_county_health_2025.xlsx"
    )

if not ACS_FILE.exists():
    raise FileNotFoundError(
        "ACS dataset not found. Run scripts/download_acs_data.py first."
    )


# 1. Load County Health Rankings data

select_df = pd.read_excel(HEALTH_FILE, sheet_name="Select Measure Data", header=1)
additional_df = pd.read_excel(HEALTH_FILE, sheet_name="Additional Measure Data", header=1)

select_df.columns = [str(col).strip() for col in select_df.columns]
additional_df.columns = [str(col).strip() for col in additional_df.columns]

select_df["FIPS"] = pd.to_numeric(select_df["FIPS"], errors="coerce")
additional_df["FIPS"] = pd.to_numeric(additional_df["FIPS"], errors="coerce")

select_df = select_df[(select_df["County"].notna()) & (select_df["FIPS"] != 42000)]
additional_df = additional_df[(additional_df["County"].notna()) & (additional_df["FIPS"] != 42000)]

health_df = select_df.merge(
    additional_df.drop(columns=["State", "County"], errors="ignore"),
    on="FIPS",
    how="left",
    suffixes=("", "_additional")
)


# 2. Load ACS income and poverty data

acs_df = pd.read_csv(ACS_FILE)

acs_df["year"] = pd.to_datetime(acs_df["year"], errors="coerce")
acs_df["county_1"] = pd.to_numeric(acs_df["county_1"], errors="coerce")

# ACS county_1 is the county code, so 42000 + county code creates PA county FIPS.
acs_df["FIPS"] = 42000 + acs_df["county_1"]

latest_year = acs_df["year"].max()
acs_latest = acs_df[acs_df["year"] == latest_year].copy()

acs_latest["acs_population"] = to_number(acs_latest["population"])
acs_latest["acs_pop_below_poverty"] = to_number(acs_latest["pop_belowpoverty"])
acs_latest["acs_poverty_rate"] = (
    acs_latest["acs_pop_below_poverty"] / acs_latest["acs_population"] * 100
)

acs_latest["acs_median_household_income"] = to_number(
    acs_latest["median_household_income"]
)
acs_latest["acs_per_capita_income"] = to_number(
    acs_latest["per_capita_income"]
)

acs_features = acs_latest[
    [
        "FIPS",
        "acs_population",
        "acs_pop_below_poverty",
        "acs_poverty_rate",
        "acs_median_household_income",
        "acs_per_capita_income",
    ]
]


# 3. Merge both data sources

df = health_df.merge(
    acs_features,
    on="FIPS",
    how="left"
)

model_df = pd.DataFrame()
model_df["FIPS"] = df["FIPS"]
model_df["County"] = df["County"]


# 4. Select and clean model features

health_feature_map = {
    "life_expectancy": ["Life Expectancy"],
    "exercise_access": ["% With Access to Exercise Opportunities"],
    "food_environment_index": ["Food Environment Index"],
    "some_college": ["% Some College"],
    "high_school_completion": ["% Completed High School"],
    "broadband_access": ["% Households with Broadband Access"],
    "social_association_rate": ["Social Association Rate"],

    "unemployment_rate": ["% Unemployed"],
    "children_poverty": ["% Children in Poverty"],
    "severe_housing_problems": ["% Severe Housing Problems"],
    "uninsured_rate": ["% Uninsured"],
    "poor_or_fair_health": ["% Fair or Poor Health"],
    "poor_physical_health_days": ["Average Number of Physically Unhealthy Days"],
    "poor_mental_health_days": ["Average Number of Mentally Unhealthy Days"],
    "income_inequality": ["Income Ratio"],
    "child_care_cost_burden": ["% Household Income Required for Child Care Expenses"],

    "rural_percent": ["% Rural"],
}

missing_features = []

for new_name, possible_names in health_feature_map.items():
    actual_col = find_column(df, possible_names)

    if actual_col is None:
        missing_features.append(new_name)
    else:
        model_df[new_name] = to_number(df[actual_col])

# ACS features from second dataset
model_df["median_household_income"] = to_number(df["acs_median_household_income"])
model_df["per_capita_income"] = to_number(df["acs_per_capita_income"])
model_df["poverty_rate"] = to_number(df["acs_poverty_rate"])
model_df["population"] = to_number(df["acs_population"])

# Fill missing numeric values with medians.
numeric_cols = model_df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if col != "FIPS":
        model_df[col] = model_df[col].fillna(model_df[col].median())


# 5. Feature engineering: create project-defined livability score

positive_features = [
    "median_household_income",
    "per_capita_income",
    "life_expectancy",
    "exercise_access",
    "food_environment_index",
    "some_college",
    "high_school_completion",
    "broadband_access",
    "social_association_rate",
]

negative_features = [
    "poverty_rate",
    "unemployment_rate",
    "children_poverty",
    "severe_housing_problems",
    "uninsured_rate",
    "poor_or_fair_health",
    "poor_physical_health_days",
    "poor_mental_health_days",
    "income_inequality",
    "child_care_cost_burden",
]

score_df = pd.DataFrame(index=model_df.index)

for col in positive_features:
    if col in model_df.columns:
        score_df[col + "_score"] = minmax_score(model_df[col], higher_is_better=True)

for col in negative_features:
    if col in model_df.columns:
        score_df[col + "_score"] = minmax_score(model_df[col], higher_is_better=False)


def average_available(columns):
    available = [col for col in columns if col in score_df.columns]
    if not available:
        return pd.Series(50, index=model_df.index)
    return score_df[available].mean(axis=1)


model_df["economic_score"] = average_available([
    "median_household_income_score",
    "per_capita_income_score",
    "poverty_rate_score",
    "unemployment_rate_score",
    "income_inequality_score",
])

model_df["health_score"] = average_available([
    "life_expectancy_score",
    "poor_or_fair_health_score",
    "poor_physical_health_days_score",
    "poor_mental_health_days_score",
    "uninsured_rate_score",
])

model_df["education_score"] = average_available([
    "some_college_score",
    "high_school_completion_score",
])

model_df["community_access_score"] = average_available([
    "exercise_access_score",
    "food_environment_index_score",
    "broadband_access_score",
    "social_association_rate_score",
])

model_df["housing_family_score"] = average_available([
    "severe_housing_problems_score",
    "child_care_cost_burden_score",
    "children_poverty_score",
])

model_df["livability_score"] = (
    0.30 * model_df["economic_score"]
    + 0.25 * model_df["health_score"]
    + 0.20 * model_df["education_score"]
    + 0.15 * model_df["community_access_score"]
    + 0.10 * model_df["housing_family_score"]
)


# 6. Train and compare ML models

feature_columns = [
    "median_household_income",
    "per_capita_income",
    "poverty_rate",
    "population",
    "life_expectancy",
    "exercise_access",
    "food_environment_index",
    "some_college",
    "high_school_completion",
    "broadband_access",
    "social_association_rate",
    "unemployment_rate",
    "children_poverty",
    "severe_housing_problems",
    "uninsured_rate",
    "poor_or_fair_health",
    "poor_physical_health_days",
    "poor_mental_health_days",
    "income_inequality",
    "child_care_cost_burden",
    "rural_percent",
]

feature_columns = [col for col in feature_columns if col in model_df.columns]

X = model_df[feature_columns]
y = model_df["livability_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

models_to_test = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree Regressor", DecisionTreeRegressor(random_state=42)),
    ("Random Forest Regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
]

results = []

for name, model in models_to_test:
    result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    results.append(result)

comparison_df = pd.DataFrame([
    {
        "model_name": result["model_name"],
        "MAE": result["MAE"],
        "RMSE": result["RMSE"],
        "R2": result["R2"],
    }
    for result in results
]).sort_values("RMSE")

# Select the best model based on lowest RMSE.
# In this project, Linear Regression performs best because the livability score
# is created from weighted numeric feature engineering.
best_model_name = comparison_df.iloc[0]["model_name"]
final_model_result = [r for r in results if r["model_name"] == best_model_name][0]
final_model = final_model_result["model"]

# Refit final model on all data for final app predictions.
final_model.fit(X, y)

model_df["predicted_livability_score"] = final_model.predict(X)
model_df["rank"] = model_df["predicted_livability_score"].rank(
    ascending=False,
    method="min"
).astype(int)


def category(score):
    if score >= 75:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 45:
        return "Average"
    else:
        return "Needs Improvement"


model_df["category"] = model_df["predicted_livability_score"].apply(category)

ranking_df = model_df.sort_values("rank")

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False)


# 7. Save all outputs

model_df.to_csv(PROCESSED_DIR / "cleaned_county_data.csv", index=False)
ranking_df.to_csv(PROCESSED_DIR / "county_rankings.csv", index=False)
importance_df.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)
comparison_df.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

joblib.dump(final_model, MODEL_DIR / "livability_model.pkl")
joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")
joblib.dump(best_model_name, MODEL_DIR / "model_name.pkl")

with open(PROCESSED_DIR / "metrics.txt", "w", encoding="utf-8") as f:
    f.write("Model Training Summary\n")
    f.write("=" * 50 + "\n\n")

    f.write("Data Sources Used:\n")
    f.write("1. 2025 Pennsylvania County Health Rankings Excel Workbook\n")
    f.write("2. PA Open Data ACS Income and Poverty County Dataset\n\n")

    f.write(f"Latest ACS Year Used: {latest_year.date()}\n")
    f.write(f"Number of counties: {len(model_df)}\n")
    f.write(f"Number of features: {len(feature_columns)}\n\n")

    f.write("Features used:\n")
    for feature in feature_columns:
        f.write(f"- {feature}\n")

    f.write("\nModel Comparison:\n")
    f.write(comparison_df.to_string(index=False))

    f.write(f"\n\nFinal Model Selected: {best_model_name}\n")
    f.write(f"MAE: {final_model_result['MAE']:.4f}\n")
    f.write(f"RMSE: {final_model_result['RMSE']:.4f}\n")
    f.write(f"R2 Score: {final_model_result['R2']:.4f}\n\n")

    f.write("Missing features:\n")
    f.write(str(missing_features))

    f.write("\n\nTop 10 counties:\n")
    f.write(
        ranking_df[
            ["rank", "County", "predicted_livability_score", "category"]
        ].head(10).to_string(index=False)
    )

    f.write("\n\nFeature importance:\n")
    f.write(importance_df.to_string(index=False))

print("Training complete.")
print()
print("Latest ACS year used:", latest_year.date())
print()
print("Model comparison:")
print(comparison_df)
print()
print(f"Final model: {best_model_name}")
print(f"MAE: {final_model_result['MAE']:.4f}")
print(f"RMSE: {final_model_result['RMSE']:.4f}")
print(f"R2 Score: {final_model_result['R2']:.4f}")
print()
print("Top 10 counties:")
print(ranking_df[["rank", "County", "predicted_livability_score", "category"]].head(10))
print()
print("Files saved:")
print("- data/processed/cleaned_county_data.csv")
print("- data/processed/county_rankings.csv")
print("- data/processed/feature_importance.csv")
print("- data/processed/model_comparison.csv")
print("- data/processed/metrics.txt")
print("- model/livability_model.pkl")
print("- model/feature_columns.pkl")