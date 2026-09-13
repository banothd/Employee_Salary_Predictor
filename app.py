import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline
with open("salary_prediction_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Page settings
st.set_page_config(
    page_title=" Employee Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# Title
st.title("💰 Employee Salary Predictor")
st.caption("Enter employee details to predict annual salary.")

# Get preprocessor
preprocessor = model.named_steps["preprocessor"]

feature_columns = list(preprocessor.feature_names_in_)

num_cols = list(preprocessor.transformers_[0][2])
cat_cols = list(preprocessor.transformers_[1][2])
ordinal_cols = list(preprocessor.transformers_[2][2])

# --------------------------------------------------
# GET ENCODERS / DEFAULT VALUES
# --------------------------------------------------

# Numerical defaults
num_pipeline = preprocessor.named_transformers_["num"]

num_imputer = num_pipeline.named_steps["imputer"]
num_defaults = dict(zip(num_cols, num_imputer.statistics_))

# Categorical information
cat_pipeline = preprocessor.named_transformers_["cat"]
cat_imputer = cat_pipeline.named_steps["imputer"]
cat_encoder = cat_pipeline.named_steps["encoder"]

cat_defaults = dict(zip(cat_cols, cat_imputer.statistics_))

# Ordinal information
ordinal_pipeline = preprocessor.named_transformers_["Ordinal Transformer"]

if hasattr(ordinal_pipeline, "named_steps"):

    ordinal_imputer = None
    ordinal_encoder = None

    for step in ordinal_pipeline.named_steps.values():

        if hasattr(step, "statistics_"):
            ordinal_imputer = step

        if hasattr(step, "categories_"):
            ordinal_encoder = step

else:
    ordinal_imputer = None
    ordinal_encoder = ordinal_pipeline

if ordinal_imputer is not None:
    ordinal_defaults = dict(
        zip(ordinal_cols, ordinal_imputer.statistics_)
    )
else:
    ordinal_defaults = {
        col: ordinal_encoder.categories_[i][0]
        for i, col in enumerate(ordinal_cols)
    }


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

st.subheader("👤 Employee Information")

col1, col2 = st.columns(2)

input_data = {}

# Years Experience
with col1:
    input_data["Years_Experience"] = st.number_input(
        "Years Experience",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=1.0
    )

# Performance Rating
with col2:
    input_data["Performance_Rating"] = st.number_input(
        "Performance Rating",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=1.0
    )

# Skills Count
with col1:
    input_data["Skills_Count"] = st.number_input(
        "Skills Count",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

# Education Level
education_index = ordinal_cols.index("Education_Level")

with col2:
    input_data["Education_Level"] = st.selectbox(
        "Education Level",
        list(ordinal_encoder.categories_[education_index]),
        key="education"
    )

# Job Level
job_index = ordinal_cols.index("Job_Level")

with col1:
    input_data["Job_Level"] = st.selectbox(
        "Job Level",
        list(ordinal_encoder.categories_[job_index]),
        key="job_level"
    )

# Department
department_index = cat_cols.index("Department")

with col2:
    input_data["Department"] = st.selectbox(
        "Department",
        list(cat_encoder.categories_[department_index]),
        key="department"
    )


# --------------------------------------------------
# AUTOMATICALLY FILL REMAINING FEATURES
# --------------------------------------------------

for col in num_cols:

    if col not in input_data:
        input_data[col] = num_defaults[col]

for col in cat_cols:

    if col not in input_data:
        input_data[col] = cat_defaults[col]

for col in ordinal_cols:

    if col not in input_data:
        input_data[col] = ordinal_defaults[col]


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Salary",
    use_container_width=True
):

    input_df = pd.DataFrame([input_data])

    # Correct feature order
    input_df = input_df[feature_columns]

    # Prediction
    prediction = model.predict(input_df)[0]

    st.success("Prediction Successful!")

    st.metric(
        "💰 Predicted Annual Salary",
        f"₹{prediction:,.0f}"
    )