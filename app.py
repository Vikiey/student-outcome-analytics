
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="Student Outcome Analytics",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_outcome_analytics_data.csv"
MODEL_PATH = BASE_DIR / "student_outcome_logistic_model.joblib"

CATEGORICAL_FEATURES = [
    "Gender", "Marital status", "Application mode", "Course",
    "Previous qualification", "Debtor", "Tuition fees up to date",
    "Scholarship holder"
]

NUMERICAL_FEATURES = [
    "Age at enrollment", "Unemployment rate",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)"
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


CATEGORY_LABELS = {
    "Gender": {
        0: "Female",
        1: "Male"
    },
    "Debtor": {
        0: "No",
        1: "Yes"
    },
    "Tuition fees up to date": {
        0: "No",
        1: "Yes"
    },
    "Scholarship holder": {
        0: "No",
        1: "Yes"
    },
    "Marital status": {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto union",
        6: "Legally separated"
    },
    "Course": {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (evening)"
    },
    "Previous qualification": {
        1: "Secondary education",
        2: "Higher education - bachelor's degree",
        3: "Higher education - degree",
        4: "Higher education - master's",
        5: "Higher education - doctorate",
        6: "Frequency of higher education",
        9: "12th year not completed",
        10: "11th year not completed",
        12: "Other - 11th year",
        14: "10th year",
        15: "10th year not completed",
        19: "Basic education 3rd cycle",
        38: "Basic education 2nd cycle",
        39: "Technological specialization course",
        40: "Higher education degree (1st cycle)",
        42: "Professional higher technical course",
        43: "Higher education master's (2nd cycle)"
    },
    "Application mode": {
        1: "1st phase - general contingent",
        2: "Ordinance No. 612/93",
        5: "1st phase - special contingent (Azores)",
        7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99",
        15: "International student (bachelor)",
        16: "1st phase - special contingent (Madeira)",
        17: "2nd phase - general contingent",
        18: "3rd phase - general contingent",
        26: "Ordinance No. 533-A/99 (item b2)",
        27: "Ordinance No. 533-A/99 (item b3)",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course",
        53: "Short cycle diploma holders",
        57: "International change of institution/course"
    }
}

# LOAD FILES
@st.cache_data
def load_data(path):
    return pd.read_csv(path, sep=";")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data(show_spinner="Training and evaluating models...")
def evaluate_models(data):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix
    )

    # Prepare features and target
    X = data[FEATURES].copy()
    y = data["Target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Keep the class order consistent across confusion matrices
    class_labels = ["Dropout", "Enrolled", "Graduate"]

    # Define the three models used in the notebook
    model_definitions = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    }

    results = []
    reports = {}
    matrices = {}

    for name, classifier in model_definitions.items():
        # Create an independent preprocessing pipeline for each model
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    CATEGORICAL_FEATURES
                ),
                (
                    "numerical",
                    StandardScaler(),
                    NUMERICAL_FEATURES
                )
            ]
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", classifier)
        ])

        # Train using only the training data
        pipeline.fit(X_train, y_train)

        # Evaluate on the held-out test data
        predictions = pipeline.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Macro F1": f1_score(
                y_test, predictions,
                average="macro", zero_division=0
            ),
            "Weighted F1": f1_score(
                y_test, predictions,
                average="weighted", zero_division=0
            ),
            "Macro Precision": precision_score(
                y_test, predictions,
                average="macro", zero_division=0
            ),
            "Macro Recall": recall_score(
                y_test, predictions,
                average="macro", zero_division=0
            )
        })

        reports[name] = pd.DataFrame(
            classification_report(
                y_test,
                predictions,
                labels=class_labels,
                output_dict=True,
                zero_division=0
            )
        ).T

        matrices[name] = pd.DataFrame(
            confusion_matrix(
                y_test,
                predictions,
                labels=class_labels
            ),
            index=class_labels,
            columns=class_labels
        )

    results_df = pd.DataFrame(results)

    return results_df, reports, matrices, len(y_test)

st.title("🎓 Student Outcome Analytics")
st.caption(
    "Explore student outcomes, evaluate machine-learning models, "
    "and estimate outcomes using first-semester information."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Saved model not found: {MODEL_PATH}")
    st.stop()

df = load_data(str(DATA_PATH))
model = load_model(str(MODEL_PATH))

required_columns = FEATURES + ["Target"]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Your CSV is missing these required columns: {missing_columns}")
    st.stop()

# SIDEBAR
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Exploratory Analysis", "Model Evaluation",
     "Predict Student Outcome"]
)

st.sidebar.caption("Project prototype • Public UCI dataset")

# OVERVIEW

if page == "Overview":
    st.title("Predicting Student Academic Outcomes")
    st.markdown(
        """
        This project uses machine learning to predict whether a student is
        likely to **Dropout**, remain **Enrolled**, or **Graduate** based on
        selected demographic, admission, financial, and first-semester
        academic information.
        """
    )

    st.info(
        "**Project objective:** Explore whether student information can "
        "help predict academic outcomes and identify patterns that may "
        "support further student-retention research."
    )

    # Project summary
    st.subheader("Project at a glance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Student records", f"{len(df):,}")

    with col2:
        st.metric("Input features", len(FEATURES))

    with col3:
        st.metric("Outcome classes", df["Target"].nunique())

    with col4:
        st.metric("Model", "Logistic Regression")

    st.divider()

    # Project purpose and scope
    st.subheader("About the project")

    left, right = st.columns(2)

    with left:
        st.markdown("#### The problem")
        st.write(
            "Student outcomes are influenced by several factors. "
            "Understanding patterns associated with dropout, continued "
            "enrolment, and graduation may help institutions investigate "
            "where additional student support could be useful."
        )

        st.markdown("#### The objective")
        st.write(
            "Build and evaluate a multiclass classification model that "
            "predicts one of three academic outcomes from selected "
            "student profile and first-semester features."
        )

    with right:
        st.markdown("#### The approach")
        st.write(
            "The project uses data preparation, categorical encoding, "
            "numerical scaling, model training, evaluation, and prediction. "
            "Logistic Regression is the current model used in the app."
        )

        st.markdown("#### Intended use")
        st.write(
            "Explore predictions and patterns for learning and research. "
            "The model is a prototype and should not independently "
            "determine decisions about individual students."
        )

    st.divider()

    st.subheader("Data Source and Scope")

    st.markdown("""
    **Dataset:** [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

    **Source:** UCI Machine Learning Repository  
    **Dataset authors:** Valentim Realinho, Mónica Vieira Martins, Jorge Machado, and Luís Baptista
    **Dataset size:** 4,424 student records and 36 features  
    **Prediction task:** Three-class classification — Dropout, Enrolled, or Graduate.

    ### Scope of this project

    This project uses demographic, socioeconomic, enrollment, and
    first-semester academic information to estimate a student's
    academic outcome.

    The prediction model uses 16 selected features, including
    student demographic details, enrollment information, financial
    indicators, and first-semester academic performance.

    Second-semester information is excluded from the model to keep
    the prediction focused on information available earlier in a
    student's academic journey.

    ### Important limitation

    The dataset represents students from a particular higher
    education institution and its undergraduate programmes. The
    model's performance may differ when applied to students from
    other institutions or educational systems.
    """)

    st.caption("""
        Dataset citation: Realinho, V., Vieira Martins, M., Machado, J., 
        & Baptista, L. (2021). Predict Students' Dropout and Academic 
        Success. UCI Machine Learning Repository. 
        https://doi.org/10.24432/C5MC89

        Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & 
        Realinho, V. (2021). Early prediction of student's performance in 
        higher education: A case study. In Trends and Applications in 
        Information Systems and Technologies (Vol. 1, pp. 166–175). Springer. 
        https://doi.org/10.1007/978-3-030-72657-7_16
    """)

    # Workflow
    st.subheader("Project workflow")

    steps = st.columns(4)

    workflow = [
        ("1. Prepare", "Inspect, clean, and select features."),
        ("2. Train", "Fit the classification pipeline."),
        ("3. Evaluate", "Review accuracy, F1 scores, and errors."),
        ("4. Predict", "Estimate outcomes for a student profile."),
    ]

    for column, (heading, description) in zip(steps, workflow):
        with column:
            st.markdown(f"**{heading}**")
            st.caption(description)

    st.divider()

    # Dataset snapshot
    st.subheader("Dataset snapshot")

    outcome_counts = df["Target"].value_counts()
    outcome_percentages = (
        df["Target"].value_counts(normalize=True).mul(100)
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Outcome distribution")

        for outcome in ["Dropout", "Enrolled", "Graduate"]:
            count = int(outcome_counts.get(outcome, 0))
            percentage = float(
                outcome_percentages.get(outcome, 0)
            )

            st.write(f"**{outcome}**")
            st.progress(percentage / 100)
            st.caption(
                f"{count:,} students ({percentage:.1f}%)"
            )

    with col2:
        st.markdown("#### Distribution chart")
        st.bar_chart(outcome_counts)

    st.caption(
        "The dataset contains historical student records. "
        "Observed class proportions describe this dataset and may not "
        "represent the proportions at another institution."
    )

    st.divider()

    # Recorded model performance
    st.subheader("Current model performance")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Test accuracy", "74.46%")

    with metric2:
        st.metric("Weighted F1 score", "0.73")

    with metric3:
        st.metric("Macro F1 score", "0.65")

    st.caption(
        "These are previously recorded evaluation results, not "
        "metrics recalculated during each app run. Keep the test "
        "split and evaluation procedure consistent when comparing models."
    )

    st.markdown(
        """
        **Interpretation:** The model performs better on the Graduate
        and Dropout classes than on Enrolled. This highlights why
        class-level precision, recall, F1 scores, and the confusion
        matrix are important alongside overall accuracy.
        """
    )

    with st.expander("Preview dataset"):
        st.dataframe(df.head(16), use_container_width=True)

    st.info(
        "This prototype uses public data. Its predictions have not "
        "been validated for use with other types of learners."
    )

# EXPLORATORY ANALYSIS
elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")

    factor = st.selectbox(
        "Select a factor to investigate",
        [
            "Gender", "Debtor", "Tuition fees up to date",
            "Scholarship holder", "Course"
        ]
    )

    rates = pd.crosstab(
        df[factor], df["Target"], normalize="index"
    ).mul(100).round(2)

    st.subheader(f"Outcome distribution by {factor}")
    st.caption("Percentages within each category; these are associations, not causes.")
    st.bar_chart(rates)

    st.subheader("First-semester academic performance")

    metric = st.selectbox(
        "Choose an academic measure",
        [
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)"
        ]
    )

    summary = df.groupby("Target")[metric].agg(
        ["mean", "median", "min", "max"]
    ).round(2)

    st.dataframe(summary, use_container_width=True)

# MODEL EVALUATION

elif page == "Model Evaluation":
    st.header("Model Evaluation")

    st.write(
        "Evaluate three classification models using the same "
        "stratified train-test split. All metrics below are calculated "
        "from predictions on the held-out test set."
    )

    try:
        results, reports, matrices, test_size = evaluate_models(df)

        st.caption(
            f"Test records: {test_size:,} | "
            "Test split: 20% | Random state: 42"
        )

        # Overall performance
        st.subheader("Overall Model Performance")

        best_accuracy = results.loc[
            results["Accuracy"].idxmax(), "Model"
        ]
        best_macro_f1 = results.loc[
            results["Macro F1"].idxmax(), "Model"
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Highest Accuracy", best_accuracy)

        with col2:
            st.metric("Highest Macro F1", best_macro_f1)

        display_results = results.copy()

        metric_columns = [
            "Accuracy",
            "Macro F1",
            "Weighted F1",
            "Macro Precision",
            "Macro Recall"
        ]

        st.dataframe(
            display_results.style.format({
                column: "{:.3f}"
                for column in metric_columns
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "All metrics range from 0 to 1. Higher values indicate "
            "better performance for the corresponding metric."
        )

        st.subheader("Compare Model Performance")
        st.bar_chart(
            results.set_index("Model")[
                ["Accuracy", "Macro F1", "Weighted F1"]
            ]
        )

        # Per-class performance
        st.divider()
        st.subheader("Classification Report")

        selected_report = st.selectbox(
            "Select a model to inspect",
            results["Model"].tolist(),
            key="evaluation_report_model"
        )

        report_df = reports[selected_report].copy()

        report_df = report_df.loc[
            ["Dropout", "Enrolled", "Graduate",
             "macro avg", "weighted avg"],
            ["precision", "recall", "f1-score", "support"]
        ]

        st.dataframe(
            report_df.style.format({
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:.0f}"
            }),
            use_container_width=True
        )

        # Confusion matrix
        st.divider()
        st.subheader("Confusion Matrix")

        selected_matrix_model = st.selectbox(
            "Select a model for the confusion matrix",
            results["Model"].tolist(),
            key="evaluation_matrix_model"
        )

        st.write(
            "Rows represent actual outcomes; columns represent "
            "predicted outcomes. The diagonal contains correct predictions."
        )

        st.dataframe(
            matrices[selected_matrix_model],
            use_container_width=True
        )

        st.caption(
            "A large number outside the diagonal indicates cases "
            "where the model predicted the wrong outcome."
        )

        # Interpretation
        st.divider()
        st.subheader("How to Interpret the Results")

        st.markdown("""
        - **Accuracy** measures the proportion of all test records
          classified correctly.
        - **Macro F1** gives equal weight to each outcome class,
          making it useful when class performance differs.
        - **Weighted F1** averages class F1 scores while accounting
          for the number of records in each class.
        - **Recall** measures how many actual cases of a class
          the model correctly identifies.
        """)

        st.info(
            "Pay particular attention to the Enrolled class. "
            "A model can achieve reasonable overall accuracy while "
            "still missing many students in this category."
        )

        st.warning(
            "These are internal test-set results. They do not establish "
            "that the models will perform equally well at other "
            "institutions or on future student populations."
        )

    except FileNotFoundError:
        st.error(
            "A required file could not be found. Check the dataset "
            "and model file paths."
        )

    except Exception as e:
        st.error(
            "The evaluation could not be completed. Check the "
            "application logs for technical details."
        )
        st.exception(e)

#PREDICTION
elif page == "Predict Student Outcome":
    st.header("Predict Student Outcome")
    st.write(
        "Enter background and first-semester information. "
        "The model estimates one of the three dataset outcome categories."
    )

    with st.form("prediction_form"):
        st.subheader("Student background")
        inputs = {}

        cols = st.columns(2)

        for i, feature in enumerate(CATEGORICAL_FEATURES):
            with cols[i % 2]:
                codes = sorted(df[feature].dropna().unique().tolist(), key=str)
                if feature in CATEGORY_LABELS:
                    label_map = CATEGORY_LABELS[feature]

                    selected_code = st.selectbox(
                        feature,
                        options=codes,
                        format_func=lambda code: (
                             f"{label_map[code]} (code: {code})"
                             if code in label_map
                             else f"Unmapped category (code: {code})"
                        ),
                        key=f"select_{feature}"
                    )
                else:
                    selected_code = st.selectbox(feature, codes)
                inputs[feature] = selected_code

        st.subheader("First-semester information", anchor=False)
        cols = st.columns(2)

        for i, feature in enumerate(NUMERICAL_FEATURES):
            series = pd.to_numeric(df[feature], errors="coerce").dropna()
            low = float(series.min())
            high = float(series.max())
            median = float(series.median())

            with cols[i % 2]:
                if low == high:
                    inputs[feature] = st.number_input(
                        feature, value=low
                    )
                else:
                    inputs[feature] = st.number_input(
                        feature,
                        min_value=low,
                        max_value=high,
                        value=median
                    )

        submitted = st.form_submit_button(
            "Estimate student outcome",
            type="primary"
        )
        
    if submitted:
        try:
            # Prepare input
            input_df = pd.DataFrame([inputs])

            # Generate prediction and probabilities
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            # Match probabilities to the model's class order
            class_names = model.classes_

            probability_df = pd.DataFrame({
                "Outcome": class_names,
                "Probability": probabilities
            })

            probability_df["Percentage"] = (
                probability_df["Probability"] * 100
            )

            # Main result
            st.divider()
            st.subheader("Prediction Results")

            if prediction == "Dropout":
                st.error("Predicted outcome: Dropout")
                interpretation = (
                    "The model classifies this profile as Dropout. "
                    "This may justify further investigation into "
                    "potential support needs, but it does not establish "
                    "that the student will leave their programme."
                )

            elif prediction == "Enrolled":
                st.info("Predicted outcome: Enrolled")
                interpretation = (
                    "The model classifies this profile as Enrolled. "
                    "It predicts this outcome over the other classes, "
                    "but it does not guarantee continued enrollment."
                )

            elif prediction == "Graduate":
                st.success("Predicted outcome: Graduate")
                interpretation = (
                    "The model classifies this profile as Graduate. "
                    "This is a prediction based on historical data, "
                    "not a guarantee of graduation."
                )

            else:
                st.warning(
                    f"The model returned an unexpected class: {prediction}"
                )
                interpretation = (
                    "Review the model's class labels before interpreting "
                    "this result."
                )

            st.write(interpretation)

            # Probability visualization
            st.subheader("Estimated probability by outcome")

            probability_df = probability_df.sort_values(
                "Probability", ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Outcome")["Percentage"]
            )

            for _, row in probability_df.iterrows():
                st.write(
                    f"**{row['Outcome']}:** "
                    f"{row['Percentage']:.2f}%"
                )

            st.caption(
                "These percentages are model-estimated class probabilities. "
                "They are not guarantees, and their reliability depends on "
                "the model's calibration and the data used to train it."
            )

            # Input summary
            with st.expander("View the student information used"):
                readable_inputs = []

                for feature, value in inputs.items():
                        label_map = CATEGORY_LABELS.get(feature, {})

                        # Translate categorical codes into readable labels
                        if feature in CATEGORY_LABELS:
                            meaning = label_map.get(value, f"Unmapped category (code: {value})")
                        else:
                            meaning = value

                        readable_inputs.append({
                "Feature": feature,
                "Model value": value,
                "Description": meaning
                })

            readable_df = pd.DataFrame(readable_inputs)

            st.dataframe(
                readable_df,
                use_container_width=True,
                hide_index=True
            )

            # Important limitation
            st.warning(
                "This application is a prototype for educational and "
                "analytical purposes. A prediction should not be used alone "
                "to make decisions about a student. Human review and "
                "additional context are essential."
            )

        except ValueError:
            st.error(
                "The prediction could not be generated because one or "
                "more input values may be invalid or incompatible with "
                "the model. Please review the inputs and try again."
            )

        except Exception:
            st.error(
                "An unexpected error occurred while generating the "
                "prediction. Please check the application logs or "
                "contact the developer if the problem persists."
            )
