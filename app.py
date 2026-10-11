
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "laptop_price_models.joblib"
DATA_PATH = "Laptop_price.csv"

# =========================================================
# CUSTOM UI STYLING
# =========================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.18);
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    div[data-testid="stForm"] {
        border-radius: 12px;
        padding: 20px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD SAVED MODELS
# =========================================================
@st.cache_resource
def load_models(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_dataset(path, modified_time):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' was not found.")
    st.info("Run your training script and save the trained models first.")
    st.stop()

try:
    artifacts = load_models(
        MODEL_PATH,
        os.path.getmtime(MODEL_PATH)
    )

    tree_model = artifacts["decision_tree"]
    forest_model = artifacts["random_forest"]
    feature_columns = list(artifacts["feature_columns"])
    brand_encoder = artifacts.get("brand_encoder")

except Exception as e:
    st.error(f"Unable to load model artifacts: {e}")
    st.stop()


models = {
    "Decision Tree Regressor": tree_model,
    "Random Forest Regressor": forest_model,
}

# =========================================================
# OPTIONAL DATASET
# =========================================================
data = None

if os.path.exists(DATA_PATH):
    try:
        data = load_dataset(
            DATA_PATH,
            os.path.getmtime(DATA_PATH)
        )
    except Exception:
        data = None

if data is not None:
    missing_features = [
        col for col in feature_columns
        if col not in data.columns
    ]

    if missing_features:
        data = None


# =========================================================
# COMMON FUNCTIONS
# =========================================================
def prepare_features(input_df):
    """Apply the encoding expected by the saved models."""

    result = input_df[feature_columns].copy()

    # Apply the same Brand LabelEncoder used during training.
    brand_col = next(
        (c for c in feature_columns if c.lower() == "brand"),
        None
    )

    if brand_col and brand_encoder is not None:
        result[brand_col] = brand_encoder.transform(
            result[brand_col].astype(str)
        )

    # Match the numeric inputs expected by the current saved models.
    for col in feature_columns:
        if brand_col and col == brand_col and brand_encoder is not None:
            continue

        result[col] = pd.to_numeric(result[col], errors="raise")

    return result


def show_prediction_result(model_name, input_df):
    """Predict and display a formatted result."""

    prepared = prepare_features(input_df)
    prediction = float(models[model_name].predict(prepared)[0])

    st.success("Prediction completed successfully!")

    a, b = st.columns([1, 2])

    with a:
        st.metric(
            "Estimated Laptop Price",
            f"{prediction:,.2f}"
        )

    with b:
        st.caption("Selected prediction model")
        st.write(f"**{model_name}**")
        st.caption(
            "Price is expressed in the units and currency "
            "used in the original training dataset."
        )

    result = input_df.copy()
    result["Selected Model"] = model_name
    result["Predicted Price"] = prediction

    with st.expander("View prediction details", expanded=True):
        st.dataframe(result, use_container_width=True)

    st.download_button(
        "⬇️ Download Prediction CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="laptop_price_prediction.csv",
        mime="text/csv",
        key="single_download",
    )


# =========================================================
# HEADER
# =========================================================
st.title("💻 Laptop Price Prediction System")
st.caption(
    "Machine Learning | Decision Tree | Random Forest | "
    "Individual & Batch Predictions"
)

st.divider()

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.title("💻 Laptop AI")
    st.caption("Price Prediction Dashboard")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict One Laptop",
            "Batch Predictions",
            "Model Comparison",
            "Dataset Explorer",
            "About",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("**Model Status**")
    st.success("Saved models loaded")

    st.caption(f"Model file: {MODEL_PATH}")

    if data is not None:
        st.caption(f"Dataset: {DATA_PATH}")
    else:
        st.caption("Optional dataset not available")


# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":

    st.subheader("Project Dashboard")
    st.write(
        "Monitor your saved models and access laptop price "
        "prediction features from one place."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Models Available", len(models))
    c2.metric("Input Features", len(feature_columns))
    c3.metric(
        "Dataset Rows",
        f"{len(data):,}" if data is not None else "N/A"
    )
    c4.metric("Model Storage", "Joblib")

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Available ML Models")

        st.markdown("""
        **🌳 Decision Tree Regressor**

        Predicts laptop prices using tree-based decision rules.

        **🌲 Random Forest Regressor**

        Combines multiple decision trees to generate predictions.
        """)

    with right:
        st.markdown("### Application Information")

        st.write("**Model file:**", MODEL_PATH)
        st.write("**Prediction target:** Price")
        st.write("**Model training:** Performed separately")
        st.write("**Input features:**", len(feature_columns))

        st.info(
            "This application loads saved models. "
            "It does not retrain the models when the page opens."
        )

    st.markdown("### Features Used by the Model")

    st.write(", ".join(feature_columns))

    if data is None:
        st.warning(
            "Place Laptop_price.csv beside app.py to enable "
            "dataset-based input choices and Dataset Explorer."
        )


# =========================================================
# SINGLE LAPTOP PREDICTION
# =========================================================
elif page == "Predict One Laptop":

    st.subheader("🔮 Predict the Price of a Laptop")
    st.write(
        "Enter the laptop specifications, choose a model, "
        "and generate an estimated price."
    )

    if data is None:
        st.warning(
            "Laptop_price.csv is needed to populate feature choices. "
            "Place the CSV in the application folder."
        )
        st.stop()

    with st.form("single_prediction_form"):

        st.markdown("#### 1. Choose Prediction Model")

        model_name = st.selectbox(
            "Machine Learning Model",
            list(models.keys()),
        )

        st.markdown("#### 2. Enter Laptop Specifications")

        st.caption(
            "Choose categorical values from the available options "
            "or enter numeric specifications."
        )

        input_values = {}
        columns = st.columns(2)

        for i, col in enumerate(feature_columns):

            with columns[i % 2]:

                series = data[col]
                is_brand = col.lower() == "brand"

                if is_brand and brand_encoder is not None:

                    options = [
                        str(x) for x in brand_encoder.classes_
                    ]

                    input_values[col] = st.selectbox(
                        "Brand",
                        options,
                        key=f"single_{col}",
                    )

                elif (
                    pd.api.types.is_numeric_dtype(series)
                    or pd.api.types.is_bool_dtype(series)
                ):

                    values = pd.to_numeric(
                        series, errors="coerce"
                    ).dropna()

                    default = (
                        float(values.median())
                        if not values.empty else 0.0
                    )

                    minimum = (
                        float(values.min())
                        if not values.empty else 0.0
                    )

                    maximum = (
                        float(values.max())
                        if not values.empty else 100.0
                    )

                    # Avoid restrictive ranges when the dataset
                    # contains only one observed numeric value.
                    if minimum == maximum:
                        input_values[col] = st.number_input(
                            col,
                            value=default,
                            key=f"single_{col}",
                        )
                    else:
                        input_values[col] = st.number_input(
                            col,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            key=f"single_{col}",
                        )

                else:

                    options = sorted(
                        series.dropna().astype(str).unique().tolist()
                    )

                    if options:
                        input_values[col] = st.selectbox(
                            col,
                            options,
                            key=f"single_{col}",
                        )
                    else:
                        input_values[col] = st.text_input(
                            col,
                            key=f"single_{col}",
                        )

        submitted = st.form_submit_button(
            "🚀 Predict Laptop Price",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame(
            [input_values],
            columns=feature_columns,
        )

        try:
            show_prediction_result(model_name, input_df)

        except Exception as e:
            st.error(
                "Prediction failed. Verify that the saved model, "
                "feature order, encoding, and input data types "
                f"match the training setup. Details: {e}"
            )


# =========================================================
# BATCH PREDICTIONS
# =========================================================
elif page == "Batch Predictions":

    st.subheader("📁 Batch Laptop Price Prediction")

    st.write(
        "Upload a CSV containing multiple laptops and predict "
        "their prices in one operation."
    )

    st.markdown("#### Required Input Columns")

    st.code(", ".join(feature_columns))

    st.caption(
        "Use the same feature names and data types as the "
        "original training dataset. The Price column is not required."
    )

    batch_file = st.file_uploader(
        "Upload Laptop CSV",
        type=["csv"],
        key="batch_upload",
    )

    model_name = st.selectbox(
        "Choose prediction model",
        list(models.keys()),
        key="batch_model",
    )

    if batch_file is not None:

        try:
            batch_df = pd.read_csv(batch_file)
            batch_df.columns = (
                batch_df.columns.astype(str).str.strip()
            )

            missing = [
                col for col in feature_columns
                if col not in batch_df.columns
            ]

            if missing:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing)
                )

            elif batch_df.empty:
                st.warning("The uploaded CSV contains no rows.")

            else:
                st.markdown("#### File Preview")

                st.dataframe(
                    batch_df.head(10),
                    use_container_width=True,
                )

                st.info(
                    f"Rows available for prediction: {len(batch_df):,}"
                )

                if st.button(
                    "🚀 Run Batch Predictions",
                    type="primary",
                ):
                    try:
                        batch_features = prepare_features(
                            batch_df[feature_columns].copy()
                        )

                        predictions = models[model_name].predict(
                            batch_features
                        )

                        output = batch_df.copy()
                        output["Predicted Price"] = predictions

                        st.session_state["batch_output"] = output
                        st.session_state["batch_model_used"] = model_name
                        st.session_state["batch_source"] = batch_file.name

                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

                if "batch_output" in st.session_state:

                    output = st.session_state["batch_output"]

                    st.divider()
                    st.markdown("### Batch Prediction Results")

                    r1, r2, r3 = st.columns(3)

                    r1.metric("Predictions", f"{len(output):,}")
                    r2.metric(
                        "Average Predicted Price",
                        f"{output['Predicted Price'].mean():,.2f}",
                    )
                    r3.metric(
                        "Prediction Model",
                        st.session_state["batch_model_used"],
                    )

                    st.dataframe(
                        output,
                        use_container_width=True,
                    )

                    st.download_button(
                        "⬇️ Download Batch Results",
                        data=output.to_csv(index=False).encode("utf-8"),
                        file_name="laptop_batch_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Unable to read the uploaded CSV: {e}")


# =========================================================
# MODEL COMPARISON
# =========================================================
elif page == "Model Comparison":

    st.subheader("📊 Model Comparison")
    st.write(
        "Compare the two saved regression models and review "
        "their evaluation results."
    )

    # Metrics must come from a genuine evaluation performed
    # on held-out test data, not training data.
    metrics = artifacts.get("metrics")

    if isinstance(metrics, dict) and metrics:

        metric_df = pd.DataFrame(metrics).T

        st.markdown("### Saved Evaluation Metrics")
        st.dataframe(
            metric_df.style.format("{:.4f}"),
            use_container_width=True,
        )

        available_metrics = [
            col for col in ["MAE", "MSE", "RMSE", "R²"]
            if col in metric_df.columns
        ]

        if available_metrics:

            selected_metric = st.selectbox(
                "Select evaluation metric",
                available_metrics,
            )

            fig, ax = plt.subplots(figsize=(9, 4))

            metric_df[selected_metric].plot(
                kind="bar",
                ax=ax,
            )

            ax.set_title(f"Model Comparison: {selected_metric}")
            ax.set_ylabel(selected_metric)
            ax.set_xlabel("Model")
            ax.tick_params(axis="x", rotation=10)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.download_button(
            "⬇️ Download Evaluation Metrics",
            data=metric_df.to_csv().encode("utf-8"),
            file_name="model_comparison_metrics.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Evaluation metrics were not found in the Joblib artifact. "
            "The saved models are available for prediction, but reliable "
            "MAE, MSE, RMSE and R² values cannot be calculated without "
            "held-out test data and the corresponding true prices."
        )

    st.markdown("### Available Models")

    for name in models:
        st.write(f"✓ {name}")

    st.markdown("### Feature Configuration")

    st.write(feature_columns)

    # Optional saved actual and predicted values.
    evaluation = artifacts.get("evaluation")

    if isinstance(evaluation, dict):
        actual = evaluation.get("y_test")
        predictions = evaluation.get("predictions", {})

        if actual is not None and isinstance(predictions, dict):

            selected = st.selectbox(
                "Model for actual vs predicted chart",
                list(predictions.keys()),
            )

            pred = np.asarray(predictions[selected])
            actual = np.asarray(actual)

            if len(actual) == len(pred) and len(actual) > 0:

                fig, ax = plt.subplots(figsize=(7, 5))

                ax.scatter(actual, pred, alpha=0.65)

                low = min(actual.min(), pred.min())
                high = max(actual.max(), pred.max())

                ax.plot(
                    [low, high],
                    [low, high],
                    linestyle="--",
                )

                ax.set_xlabel("Actual Price")
                ax.set_ylabel("Predicted Price")
                ax.set_title(f"Actual vs Predicted: {selected}")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                residuals = actual - pred

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(residuals, bins=25)
                ax.set_xlabel("Actual − Predicted")
                ax.set_ylabel("Frequency")
                ax.set_title("Prediction Residuals")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


# =========================================================
# DATASET EXPLORER
# =========================================================
elif page == "Dataset Explorer":

    st.subheader("🔍 Dataset Explorer")

    if data is None:
        st.warning(
            "Dataset Explorer requires Laptop_price.csv "
            "in the application folder."
        )
        st.stop()

    st.write(
        "Explore the dataset used to populate prediction inputs. "
        "This page does not independently evaluate model accuracy."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Rows", f"{len(data):,}")
    c2.metric("Total Columns", f"{len(data.columns):,}")
    c3.metric("Missing Values", f"{int(data.isna().sum().sum()):,}")

    st.divider()

    st.markdown("### Dataset Preview")

    st.dataframe(
        data.head(50),
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Column Data Types")

        dtype_df = data.dtypes.astype(str).rename(
            "Data Type"
        ).to_frame()

        st.dataframe(dtype_df, use_container_width=True)

    with right:
        st.markdown("### Missing Values")

        missing_df = data.isna().sum().rename(
            "Missing Count"
        ).to_frame()

        st.dataframe(missing_df, use_container_width=True)

    if "Price" in data.columns:

        prices = pd.to_numeric(
            data["Price"], errors="coerce"
        ).dropna()

        if not prices.empty:

            st.markdown("### Price Distribution")

            fig, ax = plt.subplots(figsize=(9, 4))

            ax.hist(prices, bins=30)
            ax.set_xlabel("Price")
            ax.set_ylabel("Number of Laptops")
            ax.set_title("Laptop Price Distribution")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            c1, c2, c3 = st.columns(3)

            c1.metric("Minimum Price", f"{prices.min():,.2f}")
            c2.metric("Average Price", f"{prices.mean():,.2f}")
            c3.metric("Maximum Price", f"{prices.max():,.2f}")

    st.download_button(
        "⬇️ Download Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="laptop_price_dataset.csv",
        mime="text/csv",
    )


# =========================================================
# ABOUT
# =========================================================
elif page == "About":

    st.subheader("ℹ️ About This Application")

    st.markdown("""
    ### Laptop Price Prediction System

    This application estimates laptop prices using two trained
    machine learning regression models.

    **Machine Learning Algorithms**
    - Decision Tree Regressor
    - Random Forest Regressor

    **Application Features**
    - Load saved models with Joblib
    - Individual laptop price prediction
    - Batch prediction from CSV
    - Model comparison and saved evaluation metrics
    - Dataset exploration and price distribution
    - Download prediction results

    **Technology Stack**
    - Python
    - Streamlit
    - Pandas and NumPy
    - Scikit-learn
    - Matplotlib
    - Joblib

    The models are trained and saved separately from this interface.
    Model accuracy should be evaluated on held-out test data before
    using predictions for real purchasing decisions.
    """)

    st.divider()

    st.markdown("### Model Artifact")

    st.code(MODEL_PATH)

    st.markdown("### Expected Feature Columns")

    for feature in feature_columns:
        st.write(f"- {feature}")
