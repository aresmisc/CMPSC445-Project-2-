from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "county_rankings.csv"
MODEL_FILE = BASE_DIR / "model" / "livability_model.pkl"
FEATURE_COLUMNS_FILE = BASE_DIR / "model" / "feature_columns.pkl"
MODEL_NAME_FILE = BASE_DIR / "model" / "model_name.pkl"
IMPORTANCE_FILE = BASE_DIR / "data" / "processed" / "feature_importance.csv"
METRICS_FILE = BASE_DIR / "data" / "processed" / "metrics.txt"
MODEL_COMPARISON_FILE = BASE_DIR / "data" / "processed" / "model_comparison.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_feature_importance():
    if IMPORTANCE_FILE.exists():
        return pd.read_csv(IMPORTANCE_FILE)
    return pd.DataFrame()


@st.cache_data
def load_model_comparison():
    if MODEL_COMPARISON_FILE.exists():
        return pd.read_csv(MODEL_COMPARISON_FILE)
    return pd.DataFrame()


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURE_COLUMNS_FILE)

    if MODEL_NAME_FILE.exists():
        model_name = joblib.load(MODEL_NAME_FILE)
    else:
        model_name = "Machine Learning Model"

    return model, feature_columns, model_name


def get_category(score):
    if score >= 75:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 45:
        return "Average"
    else:
        return "Needs Improvement"


def format_value(column, value):
    if pd.isna(value):
        return "N/A"

    money_columns = [
        "median_household_income",
        "per_capita_income",
    ]

    percent_columns = [
        "poverty_rate",
        "unemployment_rate",
        "children_poverty",
        "some_college",
        "high_school_completion",
        "broadband_access",
        "severe_housing_problems",
        "rural_percent",
        "exercise_access",
        "poor_or_fair_health",
        "uninsured_rate",
    ]

    if column in money_columns:
        return f"${value:,.0f}"

    if column in percent_columns:
        return f"{value:.2f}%"

    if column == "population":
        return f"{value:,.0f}"

    return f"{value:.2f}"


st.set_page_config(
    page_title="Best PA County Predictor",
    page_icon="🏙️",
    layout="wide"
)

st.title("Best PA County to Live In Predictor")

st.write(
    "This web app uses a trained machine learning model to predict and rank "
    "Pennsylvania counties based on livability-related indicators. The score is "
    "based on economic, health, education, housing, and community access factors."
)

if not DATA_FILE.exists() or not MODEL_FILE.exists() or not FEATURE_COLUMNS_FILE.exists():
    st.error("Model files were not found. Please run `python scripts/train_model.py` first.")
    st.stop()


df = load_data()
importance_df = load_feature_importance()
comparison_df = load_model_comparison()
model, feature_columns, model_name = load_model()

county_names = sorted(df["County"].dropna().unique())

st.sidebar.header("County Selection")
selected_county = st.sidebar.selectbox("Select a Pennsylvania County", county_names)

st.sidebar.markdown("---")
st.sidebar.write("**Project Theme:** Best in PA")
st.sidebar.write("**ML Task:** Regression")
st.sidebar.write(f"**Model:** {model_name}")
st.sidebar.write("**Output:** Livability Score")

county_row = df[df["County"] == selected_county].iloc[0]

X_county = county_row[feature_columns].to_frame().T
predicted_score = float(model.predict(X_county)[0])
category = get_category(predicted_score)

rank = int(county_row["rank"])
total_counties = len(df)

tab1, tab2, tab3, tab4 = st.tabs(
    ["County Prediction", "Rankings", "Model Details", "Project Notes"]
)

with tab1:
    st.header(f"Prediction Result: {selected_county} County")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Livability Score", f"{predicted_score:.2f}")

    with col2:
        st.metric("State Rank", f"{rank} / {total_counties}")

    with col3:
        st.metric("Category", category)

    st.subheader("County Feature Summary")

    display_names = {
        "median_household_income": "Median Household Income",
        "per_capita_income": "Per Capita Income",
        "poverty_rate": "Poverty Rate",
        "life_expectancy": "Life Expectancy",
        "unemployment_rate": "Unemployment Rate",
        "children_poverty": "Children in Poverty",
        "some_college": "Some College",
        "high_school_completion": "High School Completion",
        "broadband_access": "Broadband Access",
        "severe_housing_problems": "Severe Housing Problems",
        "population": "Population",
        "rural_percent": "Rural Population",
        "exercise_access": "Access to Exercise Opportunities",
        "food_environment_index": "Food Environment Index",
        "poor_or_fair_health": "Poor or Fair Health",
        "uninsured_rate": "Uninsured Rate",
        "income_inequality": "Income Inequality Ratio",
    }

    summary_rows = []

    for column, label in display_names.items():
        if column in df.columns:
            summary_rows.append({
                "Feature": label,
                "Value": format_value(column, county_row[column])
            })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.info(
        "The score is not an official government ranking. It is a project-defined "
        "machine learning score based on selected public county-level data."
    )

with tab2:
    st.header("Pennsylvania County Rankings")

    ranking_table = df.sort_values("rank")[
        ["rank", "County", "predicted_livability_score", "category"]
    ].copy()

    ranking_table = ranking_table.rename(columns={
        "rank": "Rank",
        "County": "County",
        "predicted_livability_score": "Predicted Livability Score",
        "category": "Category"
    })

    ranking_table["Predicted Livability Score"] = ranking_table[
        "Predicted Livability Score"
    ].round(2)

    st.subheader("Top 10 Counties")
    st.dataframe(ranking_table.head(10), use_container_width=True, hide_index=True)

    st.subheader("All Counties")
    st.dataframe(ranking_table, use_container_width=True, hide_index=True)

    csv = ranking_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download County Rankings as CSV",
        data=csv,
        file_name="pa_county_rankings.csv",
        mime="text/csv"
    )

with tab3:
    st.header("Model Details")

    st.write(f"**Model Type:** {model_name}")
    st.write("**Prediction Target:** Livability Score")
    st.write("**Number of Counties:**", len(df))
    st.write("**Number of Model Features:**", len(feature_columns))

    st.subheader("Model Comparison")

    if not comparison_df.empty:
        display_comparison = comparison_df.copy()

        for col in ["MAE", "RMSE", "R2"]:
            if col in display_comparison.columns:
                display_comparison[col] = display_comparison[col].round(4)

        display_comparison = display_comparison.rename(columns={
            "model_name": "Model",
            "MAE": "MAE",
            "RMSE": "RMSE",
            "R2": "R² Score"
        })

        st.dataframe(display_comparison, use_container_width=True, hide_index=True)
    else:
        st.write("Model comparison file was not found.")

    st.subheader("Features Used by the Model")

    feature_display = pd.DataFrame({
        "Feature Column": feature_columns
    })

    st.dataframe(feature_display, use_container_width=True, hide_index=True)

    if METRICS_FILE.exists():
        st.subheader("Model Evaluation Output")
        metrics_text = METRICS_FILE.read_text(encoding="utf-8")
        st.text(metrics_text)

    if not importance_df.empty:
        st.subheader("Feature Importance / Model Explanation")

        importance_display = importance_df.copy()

        if "importance" in importance_display.columns:
            importance_display["importance"] = importance_display["importance"].round(4)

        importance_display = importance_display.rename(columns={
            "feature": "Feature",
            "importance": "Importance"
        })

        st.dataframe(importance_display, use_container_width=True, hide_index=True)

        if "Feature" in importance_display.columns and "Importance" in importance_display.columns:
            chart_df = importance_display.set_index("Feature")[["Importance"]]
            st.bar_chart(chart_df)
    else:
        st.write("Feature importance file was not found.")

with tab4:
    st.header("Project Notes")

    st.write(
        "This project follows the 'Best in PA' theme by defining the best county "
        "as a county with strong livability indicators. The model uses public "
        "Pennsylvania county-level data and creates a livability score through "
        "feature engineering."
    )

    st.write(
        "The project combines two public data sources: the 2025 Pennsylvania County "
        "Health Rankings workbook and the PA Open Data ACS Income and Poverty county "
        "dataset. These sources provide information about income, poverty, health, "
        "education, housing, broadband access, population, and community conditions."
    )

    st.write(
        "The livability score combines five main areas: economic strength, health, "
        "education, community access, and housing/family conditions. Multiple regression "
        "models were compared, and the final app uses the best-performing model based "
        "on the evaluation results."
    )

    st.write(
        f"The selected final model is **{model_name}**. This application is designed "
        "for inference. Users can select a county and immediately view the predicted "
        "score, state rank, category, and supporting county indicators."
    )