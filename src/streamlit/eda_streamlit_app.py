
# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# STREAMLIT EXPLORATORY DATA ANALYSIS DASHBOARD
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Optional statistical libraries
try:
    from scipy.stats import chi2_contingency, ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False



# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================


# Current file:
# Job Acceptance Prediction System/src/streamlit/eda_streamlit_app.py

# Go up two levels to reach project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset path
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step7_feature_engineered.csv"
)

# EDA reports folder
REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "reports"
)

# EDA graphs folder
GRAPH_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "graphs"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "status"

ENGINEERED_CATEGORICAL = [
    "experience_category",
    "academic_performance_band",
    "skills_match_level",
    "interview_performance_category"
]

ENGINEERED_SCORE = "placement_probability_score"


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Job Acceptance Prediction - EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data(file_path, modified_time):

    df = pd.read_csv(file_path)

    return df


def get_data():

    if not DATA_PATH.exists():

        st.error(
            f"Dataset not found:\n{DATA_PATH}"
        )

        st.info(
            "Please make sure step7_feature_engineered.csv "
            "exists inside data/processed."
        )

        st.stop()

    modified_time = DATA_PATH.stat().st_mtime

    try:

        df = load_data(
            str(DATA_PATH),
            modified_time
        )

        return df

    except Exception as error:

        st.error(f"Unable to load dataset: {error}")
        st.stop()


df = get_data()


# ============================================================
# 4. COMMON FEATURE IDENTIFICATION
# ============================================================

def get_numerical_features(data):

    return [
        col for col in data.select_dtypes(
            include=np.number
        ).columns
        if col != TARGET
    ]


def get_categorical_features(data):

    return [
        col for col in data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
        if col != TARGET
    ]


def get_predictor_numerical_features(data):

    features = get_numerical_features(data)

    # Exclude derived score and missing indicators
    # from multicollinearity analysis.
    features = [
        col for col in features
        if col != ENGINEERED_SCORE
        and not col.endswith("_missing_flag")
    ]

    return features


NUMERICAL_FEATURES = get_numerical_features(df)
CATEGORICAL_FEATURES = get_categorical_features(df)

if TARGET not in df.columns:
    st.error("Target column 'status' is missing.")
    st.stop()

TARGET_VALUES = df[TARGET].dropna().astype(str).unique().tolist()


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 EDA Dashboard")

st.sidebar.write(
    "Job Acceptance Prediction System"
)

st.sidebar.divider()

page = st.sidebar.selectbox(
    "Select Analysis Section",
    [
        "Project Overview",
        "Data Quality Analysis",
        "Target Analysis",
        "Univariate Numerical Analysis",
        "Univariate Categorical Analysis",
        "Bivariate Numerical Analysis",
        "Bivariate Categorical Analysis",
        "Multivariate Analysis",
        "Key Insights",
        "Reports & Graphs"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Dataset Information")

st.sidebar.metric(
    "Total Records",
    f"{len(df):,}"
)

st.sidebar.metric(
    "Total Features",
    len(df.columns)
)

st.sidebar.caption(
    f"Dataset: {DATA_PATH.name}"
)


# ============================================================
# 6. PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.title("📊 Job Acceptance Prediction System")

    st.subheader("Exploratory Data Analysis Dashboard")

    st.write(
        """
        This dashboard explores the HR Job Placement Dataset
        to understand candidate characteristics, data quality,
        placement patterns, and relationships between features
        and job placement outcomes.
        """
    )

    # KPI metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Candidates", f"{len(df):,}")

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        st.metric(
            "Numerical Features",
            len(NUMERICAL_FEATURES)
        )

    with col4:
        st.metric(
            "Categorical Features",
            len(CATEGORICAL_FEATURES)
        )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Structure")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Column Names and Data Types**")

        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

    with col2:

        st.write("**Feature Type Distribution**")

        feature_type_df = pd.DataFrame({
            "Feature Type": [
                "Numerical",
                "Categorical"
            ],
            "Count": [
                len(NUMERICAL_FEATURES),
                len(CATEGORICAL_FEATURES)
            ]
        })

        fig = px.pie(
            feature_type_df,
            names="Feature Type",
            values="Count",
            hole=0.45,
            title="Numerical vs Categorical Features"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Numerical Summary")

    st.dataframe(
        df[NUMERICAL_FEATURES].describe().T
        if NUMERICAL_FEATURES
        else pd.DataFrame(),
        use_container_width=True
    )

    st.download_button(
        "Download Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="job_acceptance_eda_dataset.csv",
        mime="text/csv"
    )


# ============================================================
# 7. DATA QUALITY ANALYSIS
# ============================================================

elif page == "Data Quality Analysis":

    st.title("🔍 Data Quality Analysis")

    total_missing = int(df.isnull().sum().sum())

    duplicate_count = int(df.duplicated().sum())

    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Missing Values", f"{total_missing:,}")

    col2.metric("Duplicate Rows", f"{duplicate_count:,}")

    col3.metric(
        "Constant Columns",
        len(constant_columns)
    )

    col4.metric(
        "Complete Rows",
        f"{df.notnull().all(axis=1).sum():,}"
    )

    st.divider()

    st.subheader("Missing Value Analysis")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing Percentage": (
            df.isnull().mean().values * 100
        )
    })

    missing_df = missing_df.sort_values(
        "Missing Count",
        ascending=False
    )

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    missing_plot = missing_df[
        missing_df["Missing Count"] > 0
    ]

    if not missing_plot.empty:

        fig = px.bar(
            missing_plot,
            x="Column",
            y="Missing Percentage",
            title="Missing Values by Feature",
            text_auto=".2f"
        )

        fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.success("No missing values found.")

    st.divider()

    st.subheader("Duplicate Analysis")

    st.write(
        f"Exact duplicate rows in the current dataset: "
        f"**{duplicate_count:,}**"
    )

    st.divider()

    st.subheader("Data Types")

    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values
        }),
        use_container_width=True
    )

    st.subheader("Constant Columns")

    if constant_columns:
        st.write(constant_columns)
    else:
        st.success("No constant columns found.")

    st.download_button(
        "Download Data Quality Report",
        data=missing_df.to_csv(index=False).encode("utf-8"),
        file_name="data_quality_report.csv",
        mime="text/csv"
    )


# ============================================================
# 8. TARGET ANALYSIS
# ============================================================

elif page == "Target Analysis":

    st.title("🎯 Target Analysis")

    st.write(
        "Understand the distribution of candidates "
        "according to their placement status."
    )

    target_counts = (
        df[TARGET]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    target_counts.columns = [
        "Status",
        "Candidate Count"
    ]

    target_counts["Percentage"] = (
        target_counts["Candidate Count"]
        / len(df)
        * 100
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            target_counts,
            x="Status",
            y="Candidate Count",
            color="Status",
            text="Candidate Count",
            title="Candidate Count by Placement Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            target_counts,
            names="Status",
            values="Candidate Count",
            hole=0.4,
            title="Placement Status Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Target Distribution Summary")

    st.dataframe(
        target_counts,
        use_container_width=True
    )

    placed_count = (
        df[TARGET].astype(str) == "Placed"
    ).sum()

    placement_rate = (
        placed_count / len(df) * 100
        if len(df) else 0
    )

    st.metric(
        "Overall Placement Rate",
        f"{placement_rate:.2f}%"
    )

    st.download_button(
        "Download Target Analysis",
        data=target_counts.to_csv(index=False).encode("utf-8"),
        file_name="target_analysis.csv",
        mime="text/csv"
    )


# ============================================================
# 9. UNIVARIATE NUMERICAL ANALYSIS
# ============================================================

elif page == "Univariate Numerical Analysis":

    st.title("📈 Univariate Numerical Analysis")

    st.write(
        "Study the distribution and statistical properties "
        "of individual numerical features."
    )

    if not NUMERICAL_FEATURES:

        st.warning("No numerical features available.")

    else:

        selected_feature = st.selectbox(
            "Select Numerical Feature",
            NUMERICAL_FEATURES
        )

        series = pd.to_numeric(
            df[selected_feature],
            errors="coerce"
        ).dropna()

        if series.empty:

            st.warning("No valid numerical values available.")

        else:

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Mean", f"{series.mean():.2f}")
            col2.metric("Median", f"{series.median():.2f}")
            col3.metric("Standard Deviation", f"{series.std():.2f}")
            col4.metric("Missing Values", int(df[selected_feature].isnull().sum()))

            col1, col2 = st.columns(2)

            with col1:

                fig = px.histogram(
                    df,
                    x=selected_feature,
                    marginal="box",
                    title=f"Distribution of {selected_feature}",
                    nbins=40
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with col2:

                fig = px.box(
                    df,
                    y=selected_feature,
                    points="outliers",
                    title=f"Boxplot of {selected_feature}"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.subheader("Statistical Summary")

            summary = series.describe().to_frame(
                name="Value"
            )

            summary.loc["Variance"] = series.var()
            summary.loc["Skewness"] = series.skew()
            summary.loc["Kurtosis"] = series.kurt()

            st.dataframe(
                summary,
                use_container_width=True
            )

            st.download_button(
                "Download Feature Summary",
                data=summary.to_csv().encode("utf-8"),
                file_name=f"{selected_feature}_summary.csv",
                mime="text/csv"
            )


# ============================================================
# 10. UNIVARIATE CATEGORICAL ANALYSIS
# ============================================================

elif page == "Univariate Categorical Analysis":

    st.title("📊 Univariate Categorical Analysis")

    st.write(
        "Understand category frequencies and proportions "
        "for individual categorical features."
    )

    categorical_options = (
        CATEGORICAL_FEATURES + [TARGET]
    )

    selected_feature = st.selectbox(
        "Select Categorical Feature",
        categorical_options
    )

    frequency = (
        df[selected_feature]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    frequency.columns = [
        "Category",
        "Count"
    ]

    frequency["Percentage"] = (
        frequency["Count"] / len(df) * 100
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            frequency,
            x="Category",
            y="Count",
            text="Count",
            title=f"{selected_feature} - Category Count"
        )

        fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            frequency,
            names="Category",
            values="Count",
            title=f"{selected_feature} - Category Percentage"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Category Frequency Table")

    st.dataframe(
        frequency,
        use_container_width=True
    )

    st.download_button(
        "Download Category Frequency",
        data=frequency.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_feature}_frequency.csv",
        mime="text/csv"
    )


# ============================================================
# 11. BIVARIATE NUMERICAL ANALYSIS
# ============================================================

elif page == "Bivariate Numerical Analysis":

    st.title("📉 Bivariate Numerical Analysis")

    st.write(
        "Compare numerical feature distributions "
        "between placed and not placed candidates."
    )

    selected_feature = st.selectbox(
        "Select Numerical Feature",
        NUMERICAL_FEATURES
    )

    plot_df = df[
        [selected_feature, TARGET]
    ].copy()

    plot_df[selected_feature] = pd.to_numeric(
        plot_df[selected_feature],
        errors="coerce"
    )

    plot_df = plot_df.dropna(
        subset=[selected_feature, TARGET]
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            plot_df,
            x=TARGET,
            y=selected_feature,
            color=TARGET,
            title=f"{selected_feature} by Placement Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.violin(
            plot_df,
            x=TARGET,
            y=selected_feature,
            color=TARGET,
            box=True,
            title=f"Distribution of {selected_feature} by Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    comparison = (
        plot_df.groupby(TARGET)[selected_feature]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    st.subheader("Group Statistics")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    if SCIPY_AVAILABLE:

        st.subheader("Welch Independent t-test")

        groups = [
            group[selected_feature].dropna().values
            for _, group in plot_df.groupby(TARGET)
        ]

        if len(groups) == 2 and all(len(g) > 1 for g in groups):

            test_result = ttest_ind(
                groups[0],
                groups[1],
                equal_var=False
            )

            st.write(
                f"**t-statistic:** {test_result.statistic:.4f}"
            )

            st.write(
                f"**p-value:** {test_result.pvalue:.6f}"
            )

            if test_result.pvalue < 0.05:

                st.info(
                    "The group means show a statistically "
                    "significant difference at the 5% level."
                )

            else:

                st.info(
                    "No statistically significant difference "
                    "was detected at the 5% level."
                )

            st.caption(
                "A statistical difference does not establish causation. "
                "Consider multiple-testing correction when comparing "
                "many features."
            )

    else:

        st.warning(
            "Install scipy to display statistical tests: "
            "python -m pip install scipy"
        )

    st.download_button(
        "Download Group Comparison",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_feature}_placement_comparison.csv",
        mime="text/csv"
    )


# ============================================================
# 12. BIVARIATE CATEGORICAL ANALYSIS
# ============================================================

elif page == "Bivariate Categorical Analysis":

    st.title("📊 Bivariate Categorical Analysis")

    st.write(
        "Compare placement outcomes across categories."
    )

    if not CATEGORICAL_FEATURES:

        st.warning("No categorical predictor features available.")

    else:

        selected_feature = st.selectbox(
            "Select Categorical Feature",
            CATEGORICAL_FEATURES
        )

        analysis_df = df[
            [selected_feature, TARGET]
        ].copy()

        analysis_df[selected_feature] = (
            analysis_df[selected_feature]
            .fillna("Missing")
            .astype(str)
        )

        analysis_df[TARGET] = (
            analysis_df[TARGET].astype(str)
        )

        cross_tab = pd.crosstab(
            analysis_df[selected_feature],
            analysis_df[TARGET]
        )

        placement_data = (
            analysis_df.assign(
                placed=(
                    analysis_df[TARGET] == "Placed"
                ).astype(int)
            )
            .groupby(selected_feature, dropna=False)
            .agg(
                total_candidates=("placed", "count"),
                placed_candidates=("placed", "sum")
            )
            .reset_index()
        )

        placement_data["placement_rate"] = (
            placement_data["placed_candidates"]
            / placement_data["total_candidates"]
            * 100
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                cross_tab,
                barmode="stack",
                title="Candidate Count by Category and Status"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.bar(
                placement_data,
                x=selected_feature,
                y="placement_rate",
                text="placement_rate",
                title="Placement Rate by Category"
            )

            fig.update_layout(xaxis_tickangle=-45)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Cross Tabulation")

        st.dataframe(
            cross_tab,
            use_container_width=True
        )

        st.subheader("Placement Rate Table")

        st.dataframe(
            placement_data,
            use_container_width=True
        )

        if SCIPY_AVAILABLE and cross_tab.shape[0] > 1 and cross_tab.shape[1] > 1:

            st.subheader("Chi-square Test of Independence")

            chi2, p_value, dof, expected = chi2_contingency(
                cross_tab
            )

            n = cross_tab.to_numpy().sum()

            denominator = min(
                cross_tab.shape[0] - 1,
                cross_tab.shape[1] - 1
            )

            cramers_v = (
                np.sqrt(chi2 / (n * denominator))
                if n > 0 and denominator > 0
                else np.nan
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Chi-square", f"{chi2:.3f}")
            col2.metric("p-value", f"{p_value:.6f}")
            col3.metric("Cramér's V", f"{cramers_v:.3f}")

            st.write(
                f"Degrees of freedom: {dof}"
            )

            st.write(
                f"Minimum expected frequency: "
                f"{expected.min():.2f}"
            )

            if p_value < 0.05:

                st.info(
                    "There is evidence of an association between "
                    "the selected category and placement status."
                )

            else:

                st.info(
                    "No statistically significant association "
                    "was detected."
                )

            st.caption(
                "Check expected cell counts. Statistical association "
                "does not imply causation."
            )


# ============================================================
# 13. MULTIVARIATE ANALYSIS
# ============================================================

elif page == "Multivariate Analysis":

    st.title("🔗 Multivariate Analysis")

    st.write(
        "Explore relationships between multiple numerical "
        "predictor features."
    )

    predictor_features = get_predictor_numerical_features(df)

    if len(predictor_features) < 2:

        st.warning(
            "At least two numerical predictor features are required."
        )

    else:

        numerical_df = df[predictor_features].apply(
            pd.to_numeric,
            errors="coerce"
        )

        correlation = numerical_df.corr()

        st.subheader("Correlation Heatmap")

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.download_button(
            "Download Correlation Matrix",
            data=correlation.to_csv().encode("utf-8"),
            file_name="correlation_matrix.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader("High Correlation Pairs")

        threshold = st.slider(
            "Correlation Threshold",
            min_value=0.50,
            max_value=1.00,
            value=0.80,
            step=0.05
        )

        high_pairs = []

        for i in range(len(correlation.columns)):

            for j in range(i + 1, len(correlation.columns)):

                value = correlation.iloc[i, j]

                if abs(value) >= threshold:

                    high_pairs.append({
                        "Feature 1": correlation.columns[i],
                        "Feature 2": correlation.columns[j],
                        "Correlation": value
                    })

        high_corr_df = pd.DataFrame(high_pairs)

        if not high_corr_df.empty:

            st.dataframe(
                high_corr_df,
                use_container_width=True
            )

        else:

            st.success(
                f"No feature pairs found above |r| >= {threshold:.2f}."
            )

        st.divider()

        st.subheader("Variance Inflation Factor (VIF)")

        if STATSMODELS_AVAILABLE:

            # Remove constant columns and rows with invalid values.
            vif_df = numerical_df.replace(
                [np.inf, -np.inf],
                np.nan
            )

            vif_df = vif_df.dropna(axis=1, how="all")

            vif_df = vif_df.loc[
                :,
                vif_df.nunique(dropna=True) > 1
            ]

            vif_df = vif_df.fillna(
                vif_df.median()
            )

            # Limit large datasets for responsive dashboard use.
            if len(vif_df) > 10000:
                vif_df = vif_df.sample(
                    10000,
                    random_state=42
                )

            if vif_df.shape[1] >= 2:

                # Standardize to improve numerical stability.
                standard_deviation = vif_df.std().replace(
                    0,
                    1
                )

                vif_scaled = (
                    (vif_df - vif_df.mean())
                    / standard_deviation
                )

                vif_results = []

                for index, column in enumerate(
                    vif_scaled.columns
                ):

                    try:

                        vif_value = variance_inflation_factor(
                            vif_scaled.values,
                            index
                        )

                    except Exception:
                        vif_value = np.nan

                    vif_results.append({
                        "Feature": column,
                        "VIF": vif_value
                    })

                vif_result_df = pd.DataFrame(
                    vif_results
                ).sort_values(
                    "VIF",
                    ascending=False
                )

                st.dataframe(
                    vif_result_df,
                    use_container_width=True
                )

                st.caption(
                    "VIF is calculated using numerical predictors only. "
                    "The derived placement readiness score and missing "
                    "indicator flags are excluded. Very high VIF values "
                    "may indicate multicollinearity."
                )

                st.download_button(
                    "Download VIF Report",
                    data=vif_result_df.to_csv(index=False).encode("utf-8"),
                    file_name="vif_analysis.csv",
                    mime="text/csv"
                )

            else:

                st.warning(
                    "Not enough usable numerical features for VIF."
                )

        else:

            st.warning(
                "Statsmodels is not installed. Run: "
                "python -m pip install statsmodels"
            )


# ============================================================
# 14. KEY INSIGHTS
# ============================================================

elif page == "Key Insights":

    st.title("💡 Key EDA Insights")

    st.write(
        "Automatically generated observations from the dataset."
    )

    total_candidates = len(df)

    placed_count = (
        df[TARGET].astype(str) == "Placed"
    ).sum()

    not_placed_count = (
        df[TARGET].astype(str) == "Not Placed"
    ).sum()

    placement_rate = (
        placed_count / total_candidates * 100
        if total_candidates else 0
    )

    st.subheader("1. Overall Placement Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Placed", f"{placed_count:,}")
    col2.metric("Not Placed", f"{not_placed_count:,}")
    col3.metric("Placement Rate", f"{placement_rate:.2f}%")

    st.divider()

    st.subheader("2. Categorical Feature Insights")

    insight_rows = []

    for feature in CATEGORICAL_FEATURES:

        temp = df[
            [feature, TARGET]
        ].copy()

        temp[feature] = (
            temp[feature].fillna("Missing").astype(str)
        )

        temp["is_placed"] = (
            temp[TARGET].astype(str) == "Placed"
        ).astype(int)

        grouped = temp.groupby(feature).agg(
            candidates=("is_placed", "count"),
            placed=("is_placed", "sum")
        )

        grouped["placement_rate"] = (
            grouped["placed"]
            / grouped["candidates"]
            * 100
        )

        # Avoid drawing conclusions from very small groups.
        eligible = grouped[
            grouped["candidates"] >= 30
        ]

        if eligible.empty:
            continue

        highest_category = eligible[
            "placement_rate"
        ].idxmax()

        lowest_category = eligible[
            "placement_rate"
        ].idxmin()

        insight_rows.append({
            "Feature": feature,
            "Highest Placement Category": str(highest_category),
            "Highest Placement Rate": eligible.loc[
                highest_category, "placement_rate"
            ],
            "Lowest Placement Category": str(lowest_category),
            "Lowest Placement Rate": eligible.loc[
                lowest_category, "placement_rate"
            ]
        })

    if insight_rows:

        st.dataframe(
            pd.DataFrame(insight_rows),
            use_container_width=True
        )

    else:

        st.info(
            "No categorical insights could be generated."
        )

    st.caption(
        "Categories with fewer than 30 candidates are excluded "
        "from this comparison to reduce unstable small-group findings."
    )

    st.divider()

    st.subheader("3. Numerical Feature Comparison")

    numerical_insights = []

    for feature in NUMERICAL_FEATURES:

        if feature == ENGINEERED_SCORE:
            continue

        placed_values = pd.to_numeric(
            df.loc[
                df[TARGET].astype(str) == "Placed",
                feature
            ],
            errors="coerce"
        ).dropna()

        not_placed_values = pd.to_numeric(
            df.loc[
                df[TARGET].astype(str) == "Not Placed",
                feature
            ],
            errors="coerce"
        ).dropna()

        if placed_values.empty or not_placed_values.empty:
            continue

        numerical_insights.append({
            "Feature": feature,
            "Placed Mean": placed_values.mean(),
            "Not Placed Mean": not_placed_values.mean(),
            "Mean Difference": (
                placed_values.mean()
                - not_placed_values.mean()
            )
        })

    if numerical_insights:

        numerical_insights_df = pd.DataFrame(
            numerical_insights
        )

        st.dataframe(
            numerical_insights_df,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical comparisons could be generated."
        )

    st.divider()

    st.subheader("4. Data Quality Observations")

    st.write(
        f"- Total missing values: {df.isnull().sum().sum():,}"
    )

    st.write(
        f"- Exact duplicate rows: {df.duplicated().sum():,}"
    )

    st.write(
        f"- Total columns: {len(df.columns)}"
    )

    st.write(
        f"- Numerical features: {len(NUMERICAL_FEATURES)}"
    )

    st.write(
        f"- Categorical features: {len(CATEGORICAL_FEATURES)}"
    )


# ============================================================
# 15. REPORTS AND GRAPHS
# ============================================================

elif page == "Reports & Graphs":

    st.title("📁 EDA Reports & Graphs")

    st.write(
        "View and download reports and graphs generated "
        "by your individual EDA scripts."
    )

    st.subheader("Generated CSV Reports")

    if REPORT_DIR.exists():

        report_files = sorted(
            REPORT_DIR.glob("*.csv")
        )

        if report_files:

            for report_file in report_files:

                with st.expander(report_file.name):

                    try:

                        report_df = pd.read_csv(
                            report_file
                        )

                        st.dataframe(
                            report_df.head(100),
                            use_container_width=True
                        )

                        st.download_button(
                            f"Download {report_file.name}",
                            data=report_file.read_bytes(),
                            file_name=report_file.name,
                            mime="text/csv",
                            key=f"download_{report_file.name}"
                        )

                    except Exception as error:

                        st.warning(
                            f"Unable to read report: {error}"
                        )

        else:

            st.info(
                "No CSV reports found. Run your EDA scripts first."
            )

    else:

        st.info("EDA report folder does not exist yet.")

    st.divider()

    st.subheader("Generated Graphs")

    if GRAPH_DIR.exists():

        graph_files = sorted([
            path
            for path in GRAPH_DIR.rglob("*")
            if path.suffix.lower() in [
                ".png", ".jpg", ".jpeg", ".webp"
            ]
        ])

        if graph_files:

            for graph_file in graph_files:

                with st.expander(
                    str(graph_file.relative_to(GRAPH_DIR))
                ):

                    st.image(
                        str(graph_file),
                        caption=graph_file.name,
                        use_container_width=True
                    )

                    st.download_button(
                        f"Download {graph_file.name}",
                        data=graph_file.read_bytes(),
                        file_name=graph_file.name,
                        mime="image/png",
                        key=f"graph_{graph_file.name}"
                    )

        else:

            st.info(
                "No saved graphs found. Run your EDA scripts first."
            )

    else:

        st.info("EDA graph folder does not exist yet.")


# ============================================================
# 16. FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Job Acceptance Prediction System"
)

st.sidebar.caption(
    "Exploratory Data Analysis Dashboard"
)

st.sidebar.caption(
    "Built using Python, Pandas, Streamlit and Plotly"
)
