
# ============================================================
# FinResilience AI
# Financial Resilience Score + About & Project Details
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

REPORT_DIR = RESULTS_DIR / "final_report"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
VALIDATION_PATH = REPORT_DIR / "final_validation_summary.json"
RESEARCH_SUMMARY_PATH = RESULTS_DIR / "research_summary.json"
REPLICATION_PATH = RESULTS_DIR / "secondary_dataset_replication.csv"

st.set_page_config(
    page_title="FinResilience AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# 2. CUSTOM STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
        color: #17324D;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        color: #17324D;
        margin-top: 1rem;
    }

    .small-note {
        font-size: 0.9rem;
        color: #64748b;
    }

    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e5eaf1;
        padding: 15px;
        border-radius: 12px;
    }

    div.stButton > button[kind="primary"] {
        border-radius: 8px;
        min-height: 2.8rem;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 3. LOAD SAVED PROJECT REPORTS
# ------------------------------------------------------------

@st.cache_data
def load_json_file(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data
def load_csv_file(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


metadata = load_json_file(str(METADATA_PATH))
validation = load_json_file(str(VALIDATION_PATH))
research_summary = load_json_file(str(RESEARCH_SUMMARY_PATH))
replication = load_csv_file(str(REPLICATION_PATH))


# ------------------------------------------------------------
# 4. FINANCIAL RESILIENCE SCORE CALCULATIONS
# ------------------------------------------------------------

def scale_score(value, full_marks_at, zero_marks_at):
    """
    Piecewise-linear scaling from 0 to 100.

    If lower values are better, full_marks_at must be less
    than zero_marks_at.
    If higher values are better, full_marks_at must be greater
    than zero_marks_at.
    """
    if not np.isfinite(value):
        return 0.0

    if full_marks_at < zero_marks_at:
        if value <= full_marks_at:
            return 100.0
        if value >= zero_marks_at:
            return 0.0

    elif full_marks_at > zero_marks_at:
        if value >= full_marks_at:
            return 100.0
        if value <= zero_marks_at:
            return 0.0

    else:
        return 0.0

    return float(
        np.clip(
            100.0
            * (value - zero_marks_at)
            / (full_marks_at - zero_marks_at),
            0,
            100,
        )
    )


def calculate_resilience_score(
    monthly_income,
    essential_expenses,
    total_expenses,
    monthly_emi,
    savings,
    total_debt,
    recent_expenses,
):
    """
    Calculate the rule-based score from the project concept.

    Weights:
        EMI burden                  25%
        Debt-to-income proxy        20%
        Savings rate                20%
        Emergency coverage          25%
        Expense volatility          10%

    Debt-to-income proxy:
        Total outstanding debt / annual income.

    This proxy is used because the concept note does not
    specify the exact time basis for the debt-to-income ratio.
    It is distinct from EMI / monthly income.
    """

    if monthly_income <= 0:
        raise ValueError("Monthly income must be greater than zero.")

    if essential_expenses <= 0:
        raise ValueError(
            "Essential monthly expenses must be greater than zero."
        )

    if total_expenses < 0:
        raise ValueError("Total expenses cannot be negative.")

    if monthly_emi < 0:
        raise ValueError("Monthly EMI cannot be negative.")

    if savings < 0 or total_debt < 0:
        raise ValueError("Savings and debt cannot be negative.")

    if essential_expenses > total_expenses:
        raise ValueError(
            "Essential expenses cannot exceed total expenses."
        )

    # 1. EMI burden: full marks at 20% or lower;
    #    zero marks at 50% or higher.
    emi_burden = monthly_emi / monthly_income
    emi_score = scale_score(emi_burden, 0.20, 0.50)

    # 2. Debt-to-income proxy: outstanding debt / annual income.
    annual_income = monthly_income * 12
    debt_to_income = total_debt / annual_income
    debt_score = scale_score(debt_to_income, 0.20, 0.60)

    # 3. Savings rate: monthly income minus total expenses,
    #    divided by monthly income.
    savings_rate = (
        monthly_income - total_expenses
    ) / monthly_income

    savings_score = scale_score(savings_rate, 0.20, 0.00)

    # 4. Emergency coverage: savings / essential monthly expenses.
    emergency_months = savings / essential_expenses
    emergency_score = scale_score(emergency_months, 6.0, 0.0)

    # 5. Expense volatility: coefficient of variation (CV).
    valid_expenses = [
        float(value)
        for value in recent_expenses
        if value is not None
        and np.isfinite(value)
        and value >= 0
    ]

    if len(valid_expenses) >= 2:
        mean_expenses = float(np.mean(valid_expenses))

        if mean_expenses > 0:
            expense_cv = float(
                np.std(valid_expenses, ddof=0) / mean_expenses
            )
            volatility_score = scale_score(
                expense_cv, 0.05, 0.30
            )
        else:
            expense_cv = np.nan
            volatility_score = 0.0
    else:
        expense_cv = np.nan
        volatility_score = 0.0

    weights = {
        "EMI burden": 0.25,
        "Debt-to-income proxy": 0.20,
        "Savings rate": 0.20,
        "Emergency coverage": 0.25,
        "Expense volatility": 0.10,
    }

    component_scores = {
        "EMI burden": emi_score,
        "Debt-to-income proxy": debt_score,
        "Savings rate": savings_score,
        "Emergency coverage": emergency_score,
        "Expense volatility": volatility_score,
    }

    # If there is insufficient expense history, exclude the
    # volatility component and proportionally rescale the other
    # weights. This avoids automatically assigning zero marks
    # for a component that could not be measured.
    if not np.isfinite(expense_cv):
        available_weight = (
            1.0 - weights["Expense volatility"]
        )

        score = sum(
            component_scores[name] * weights[name]
            for name in component_scores
            if name != "Expense volatility"
        ) / available_weight
    else:
        score = sum(
            component_scores[name] * weights[name]
            for name in component_scores
        )

    score = float(np.clip(score, 0, 100))

    if score < 40:
        band = "Fragile"
        description = (
            "Your current inputs indicate areas of financial "
            "vulnerability. Review debt obligations, expenses "
            "and emergency savings."
        )
    elif score < 70:
        band = "Watchful"
        description = (
            "Your current inputs indicate a mixed financial "
            "position. Improving savings or reducing debt burden "
            "may improve the score."
        )
    else:
        band = "Resilient"
        description = (
            "Your current inputs meet more of the score's "
            "financial-stability thresholds. Continue monitoring "
            "expenses and maintaining an emergency reserve."
        )

    return {
        "score": score,
        "band": band,
        "description": description,
        "components": component_scores,
        "emi_burden": emi_burden,
        "debt_to_income": debt_to_income,
        "savings_rate": savings_rate,
        "emergency_months": emergency_months,
        "expense_cv": expense_cv,
        "weights": weights,
    }


# ------------------------------------------------------------
# 5. SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.markdown("## FinResilience AI")
st.sidebar.caption("Financial resilience assessment")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Financial Resilience Score",
        "About & Project Details",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Academic prototype • Not a lending decision system"
)


# ------------------------------------------------------------
# 6. MAIN PAGE: FINANCIAL RESILIENCE SCORE
# ------------------------------------------------------------

if page == "Financial Resilience Score":

    st.markdown(
        '<div class="main-title">Financial Resilience Score</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Understand your financial position using income, expenses, '
        'debt, savings and emergency coverage.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This is a transparent, rule-based score. It is separate "
        "from the machine-learning credit-stress model."
    )

    with st.form("resilience_form"):

        st.markdown(
            '<div class="section-title">1. Income and monthly expenses</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            monthly_income = st.number_input(
                "Monthly income (₹)",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                help="Your total income received in a typical month.",
            )

            essential_expenses = st.number_input(
                "Essential monthly expenses (₹)",
                min_value=0.0,
                value=20000.0,
                step=500.0,
                help=(
                    "Rent, groceries, utilities, transport and "
                    "other essential living expenses."
                ),
            )

        with col2:
            total_expenses = st.number_input(
                "Total monthly expenses (₹)",
                min_value=0.0,
                value=30000.0,
                step=500.0,
                help=(
                    "All monthly expenses, including essential "
                    "and non-essential spending."
                ),
            )

            monthly_emi = st.number_input(
                "Total monthly EMI (₹)",
                min_value=0.0,
                value=8000.0,
                step=500.0,
                help="Total monthly instalment payments.",
            )

        st.markdown(
            '<div class="section-title">2. Savings and debt</div>',
            unsafe_allow_html=True,
        )

        col3, col4 = st.columns(2)

        with col3:
            savings = st.number_input(
                "Available savings (₹)",
                min_value=0.0,
                value=100000.0,
                step=5000.0,
                help=(
                    "Savings available for emergencies. "
                    "Do not include assets that cannot readily "
                    "be used to cover expenses."
                ),
            )

        with col4:
            total_debt = st.number_input(
                "Total outstanding debt (₹)",
                min_value=0.0,
                value=200000.0,
                step=5000.0,
                help=(
                    "Total remaining loan and other debt balances."
                ),
            )

        st.markdown(
            '<div class="section-title">3. Recent monthly expenses</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Enter actual expenses for up to six recent months. "
            "Use at least two months to calculate expense volatility. "
            "If fewer than six months are available, leave the "
            "remaining fields blank."
        )

        use_six_months = st.checkbox(
            "I have expense records for all six months",
            value=True,
        )

        expense_cols = st.columns(3)
        recent_expenses = []

        for index in range(6):
            with expense_cols[index % 3]:
                default_value = (
                    float(total_expenses)
                    if use_six_months or index < 2
                    else None
                )

                expense_value = st.number_input(
                    f"Month {index + 1} expenses (₹)",
                    min_value=0.0,
                    value=default_value,
                    step=500.0,
                    key=f"monthly_expense_{index}",
                    help=(
                        "Enter the total expenses for this month. "
                        "Use the same currency and monthly basis."
                    ),
                    placeholder="Leave blank if unavailable",
                )

                # For incomplete history, zero is treated as
                # missing only when the user opted out of six months.
                if not use_six_months and index >= 2:
                    recent_expenses.append(None)
                else:
                    recent_expenses.append(expense_value)

        st.divider()

        submitted = st.form_submit_button(
            "Calculate my resilience score",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            result = calculate_resilience_score(
                monthly_income=monthly_income,
                essential_expenses=essential_expenses,
                total_expenses=total_expenses,
                monthly_emi=monthly_emi,
                savings=savings,
                total_debt=total_debt,
                recent_expenses=recent_expenses,
            )

            st.divider()
            st.markdown(
                '<div class="section-title">Your result</div>',
                unsafe_allow_html=True,
            )

            result_col, band_col = st.columns([1, 1])

            with result_col:
                st.metric(
                    "Financial Resilience Score",
                    f"{result['score']:.1f} / 100",
                )
                st.progress(result["score"] / 100)

            with band_col:
                st.metric(
                    "Resilience category",
                    result["band"],
                )
                st.write(result["description"])

            st.caption(
                "The score is an educational indicator based on "
                "the entered figures and documented thresholds. "
                "It is not a credit approval or financial advice."
            )

            st.divider()
            st.markdown(
                '<div class="section-title">Score breakdown</div>',
                unsafe_allow_html=True,
            )

            component_df = pd.DataFrame(
                {
                    "Component": list(
                        result["components"].keys()
                    ),
                    "Component score": list(
                        result["components"].values()
                    ),
                    "Weight (%)": [
                        result["weights"][name] * 100
                        for name in result["components"]
                    ],
                }
            )

            st.bar_chart(
                component_df.set_index("Component")[
                    ["Component score"]
                ],
                horizontal=True,
            )

            st.dataframe(
                component_df,
                hide_index=True,
                use_container_width=True,
            )

            st.divider()
            st.markdown(
                '<div class="section-title">Your financial indicators</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "EMI / monthly income",
                f"{result['emi_burden']:.1%}",
            )

            m2.metric(
                "Debt / annual income",
                f"{result['debt_to_income']:.1%}",
            )

            m3.metric(
                "Savings rate",
                f"{result['savings_rate']:.1%}",
            )

            m4, m5 = st.columns(2)

            m4.metric(
                "Emergency coverage",
                f"{result['emergency_months']:.1f} months",
            )

            if np.isfinite(result["expense_cv"]):
                m5.metric(
                    "Expense volatility (CV)",
                    f"{result['expense_cv']:.3f}",
                )
            else:
                m5.metric(
                    "Expense volatility (CV)",
                    "Not calculated",
                )
                st.warning(
                    "Expense volatility needs at least two "
                    "available months with a positive average. "
                    "The other score components have been "
                    "rescaled to account for the missing component."
                )

            st.divider()
            st.markdown(
                '<div class="section-title">How to interpret the score</div>',
                unsafe_allow_html=True,
            )

            band_col1, band_col2, band_col3 = st.columns(3)

            with band_col1:
                st.markdown("**Fragile: 0–39**")
                st.write(
                    "The entered figures fall below more of the "
                    "score's stability thresholds."
                )

            with band_col2:
                st.markdown("**Watchful: 40–69**")
                st.write(
                    "The entered figures meet some, but not all, "
                    "of the score's stability thresholds."
                )

            with band_col3:
                st.markdown("**Resilient: 70–100**")
                st.write(
                    "The entered figures meet more of the "
                    "score's stability thresholds."
                )

            with st.expander("View calculation details"):
                st.write(
                    "**EMI burden:** monthly EMI divided by "
                    "monthly income. Full marks at 20% or below; "
                    "zero marks at 50% or above."
                )

                st.write(
                    "**Debt-to-income proxy:** total outstanding "
                    "debt divided by annual income. This is the "
                    "time-basis assumption used in this prototype."
                )

                st.write(
                    "**Savings rate:** (monthly income − total "
                    "monthly expenses) divided by monthly income."
                )

                st.write(
                    "**Emergency coverage:** available savings "
                    "divided by essential monthly expenses."
                )

                st.write(
                    "**Expense volatility:** standard deviation "
                    "of recent monthly expenses divided by their "
                    "mean (coefficient of variation)."
                )

                st.write(
                    "**Weights:** EMI burden 25%, debt-to-income "
                    "proxy 20%, savings rate 20%, emergency "
                    "coverage 25%, and expense volatility 10%."
                )

        except ValueError as exc:
            st.error(str(exc))

    else:
        st.markdown("### What contributes to your score?")

        weights_df = pd.DataFrame(
            {
                "Component": [
                    "EMI burden",
                    "Debt-to-income proxy",
                    "Savings rate",
                    "Emergency coverage",
                    "Expense volatility",
                ],
                "Weight": [25, 20, 20, 25, 10],
            }
        )

        st.bar_chart(
            weights_df.set_index("Component"),
            horizontal=True,
        )

        st.caption(
            "Enter your figures above and click Calculate my "
            "resilience score to view your personalized result."
        )


# ------------------------------------------------------------
# 7. ABOUT & PROJECT DETAILS
# ------------------------------------------------------------

elif page == "About & Project Details":

    st.markdown(
        '<div class="main-title">About FinResilience AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Project methodology, model evaluation, explainability '
        'and limitations.'
        '</div>',
        unsafe_allow_html=True,
    )

    about_tabs = st.tabs(
        [
            "Project Overview",
            "Model Evaluation",
            "SHAP Explainability",
            "Score Methodology",
            "Limitations",
        ]
    )

    # --------------------------------------------------------
    # TAB 1: PROJECT OVERVIEW
    # --------------------------------------------------------

    with about_tabs[0]:

        st.subheader("Project objective")

        st.write(
            """
            FinResilience AI investigates whether historical credit
            behaviour adds useful predictive information beyond
            conventional application characteristics when estimating
            repayment stress.

            Alongside that research experiment, the application
            provides an independent, rule-based financial-resilience
            score using user-entered financial information.
            """
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Conventional feature count", "25")
        c2.metric("Behavioural model feature count", "57")
        c3.metric("Primary metric", "PR-AUC")

        st.subheader("Model design")

        st.markdown(
            """
            **Group A — Conventional model**
            - Application and financial characteristics.
            - Evaluates the conventional-feature baseline.

            **Group B — Conventional + behavioural model**
            - Includes Group A features.
            - Adds aggregated instalment, credit-card and bureau-history
              indicators.

            **Secondary dataset**
            - Give Me Some Credit is a separate replication experiment.
            - Its target and feature space differ from Home Credit.
            """
        )

        st.info(
            "The rule-based Financial Resilience Score is not a "
            "machine-learning prediction. The two outputs use "
            "different information and should not be treated as "
            "interchangeable."
        )

    # --------------------------------------------------------
    # TAB 2: MODEL EVALUATION
    # --------------------------------------------------------

    with about_tabs[1]:

        st.subheader("Home Credit evaluation")

        if validation:
            metric_cols = st.columns(3)

            metric_cols[0].metric(
                "PR-AUC",
                f"{validation.get('pr_auc', np.nan):.4f}",
            )

            metric_cols[1].metric(
                "ROC-AUC",
                f"{validation.get('roc_auc', np.nan):.4f}",
            )

            metric_cols[2].metric(
                "Brier score",
                f"{validation.get('brier_score', np.nan):.4f}",
            )

            metric_cols2 = st.columns(3)

            metric_cols2[0].metric(
                "Precision",
                f"{validation.get('precision', np.nan):.1%}",
            )

            metric_cols2[1].metric(
                "Recall",
                f"{validation.get('recall', np.nan):.1%}",
            )

            metric_cols2[2].metric(
                "F1-score",
                f"{validation.get('f1', np.nan):.4f}",
            )

            st.caption(
                "These are the metrics recorded in the saved final "
                "validation summary. They are not recalculated here."
            )

            if "test_partition_used_for_model_tuning" in validation:
                st.write(
                    "Test partition used for model tuning: "
                    f"**{validation['test_partition_used_for_model_tuning']}**"
                )

            st.warning(
                "The recorded recall is approximately 40.5%, below "
                "the aspirational 70% target described in the project "
                "plan. Report the measured result honestly."
            )
        else:
            st.warning(
                "Final validation summary not found. Expected file: "
                "results/final_report/final_validation_summary.json"
            )

        st.subheader("Secondary dataset replication")

        if not replication.empty:
            st.dataframe(
                replication,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Secondary-dataset results come from a different "
                "target and feature space. Do not interpret the "
                "numbers as a direct same-population comparison."
            )
        else:
            st.info(
                "Secondary dataset replication CSV was not found."
            )

        st.subheader("Saved evaluation plots")

        evaluation_plots = [
            (
                "ROC and precision-recall curves",
                RESULTS_DIR / "roc_pr_curves.png",
            ),
            (
                "Calibration curves",
                RESULTS_DIR / "calibration_curves.png",
            ),
            (
                "Confusion matrix",
                REPORT_DIR / "confusion_matrix.png",
            ),
            (
                "Score distribution",
                REPORT_DIR / "score_distribution.png",
            ),
        ]

        for title, path in evaluation_plots:
            if path.exists():
                with st.expander(title):
                    st.image(
                        str(path),
                        use_container_width=True,
                    )

    # --------------------------------------------------------
    # TAB 3: SHAP EXPLAINABILITY
    # --------------------------------------------------------

    with about_tabs[2]:

        st.subheader("Explainable AI using SHAP")

        st.write(
            """
            SHAP (SHapley Additive exPlanations) estimates how
            individual features contribute to a model's output.

            Your Notebook 2 calculates SHAP values for the selected
            Group B pipeline using a background sample and an
            explanation sample. It then saves global beeswarm and
            mean-absolute-SHAP bar plots.
            """
        )

        shap_beeswarm = (
            RESULTS_DIR / "shap_global_beeswarm.png"
        )

        shap_importance = (
            RESULTS_DIR / "shap_global_importance.png"
        )

        if shap_beeswarm.exists():
            st.markdown("#### Global SHAP beeswarm plot")

            st.image(
                str(shap_beeswarm),
                use_container_width=True,
            )

            st.caption(
                "Each point represents a feature value for an "
                "applicant. SHAP values indicate the direction and "
                "magnitude of that feature's contribution to the "
                "underlying model output."
            )
        else:
            st.warning(
                "SHAP beeswarm plot not found at "
                "results/shap_global_beeswarm.png."
            )

        if shap_importance.exists():
            st.markdown("#### Global SHAP feature importance")

            st.image(
                str(shap_importance),
                use_container_width=True,
            )

            st.caption(
                "The bar plot ranks features by mean absolute SHAP "
                "value in the explanation sample."
            )
        else:
            st.warning(
                "SHAP importance plot not found at "
                "results/shap_global_importance.png."
            )

        st.markdown("#### How to interpret SHAP")

        st.markdown(
            """
            - **Positive SHAP value:** pushes the underlying model
              output toward the explained positive class.
            - **Negative SHAP value:** pushes the output away from
              the explained positive class.
            - **Larger absolute value:** indicates a larger
              contribution relative to the model's baseline output.
            - **Feature importance:** indicates model reliance in
              the analyzed sample, not proof of causation.
            """
        )

        st.warning(
            "The saved plots explain the Group B pipeline selected "
            "during Notebook 2. SHAP values are not automatically "
            "explanations of calibrated probabilities. This dashboard "
            "does not currently calculate an applicant-specific "
            "waterfall plot."
        )

        st.info(
            "The project currently includes global SHAP plots. "
            "A local waterfall plot and a plain-language explanation "
            "for an individual applicant require the matching saved "
            "explainer/preprocessing or a correctly reconstructed "
            "explanation workflow."
        )

    # --------------------------------------------------------
    # TAB 4: SCORE METHODOLOGY
    # --------------------------------------------------------

    with about_tabs[3]:

        st.subheader("Rule-based financial-resilience score")

        st.write(
            """
            The score ranges from 0 to 100. Each component is scaled
            between its full-mark and zero-mark thresholds, then
            combined using documented weights.
            """
        )

        methodology = pd.DataFrame(
            {
                "Component": [
                    "EMI burden",
                    "Debt-to-income proxy",
                    "Savings rate",
                    "Emergency coverage",
                    "Expense volatility",
                ],
                "Weight": [
                    "25%",
                    "20%",
                    "20%",
                    "25%",
                    "10%",
                ],
                "Full marks": [
                    "20% or lower",
                    "20% or lower",
                    "20% or higher",
                    "6 months or more",
                    "CV 0.05 or lower",
                ],
                "Zero marks": [
                    "50% or higher",
                    "60% or higher",
                    "0% or lower",
                    "0 months",
                    "CV 0.30 or higher",
                ],
            }
        )

        st.dataframe(
            methodology,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            **Score bands**

            - **0–39:** Fragile
            - **40–69:** Watchful
            - **70–100:** Resilient
            """
        )

        st.warning(
            "For this implementation, debt-to-income is defined as "
            "total outstanding debt divided by annual income. This is "
            "a documented proxy assumption because the concept note "
            "does not explicitly state the time basis for this ratio."
        )

        st.write(
            "If fewer than two valid months of expenses are available, "
            "expense volatility is not calculated and the remaining "
            "component weights are rescaled proportionally."
        )

    # --------------------------------------------------------
    # TAB 5: LIMITATIONS
    # --------------------------------------------------------

    with about_tabs[4]:

        st.subheader("Limitations and responsible use")

        st.markdown(
            """
            **Dataset limitations**
            - Home Credit and Give Me Some Credit represent different
              populations and use different targets.
            - The datasets may not represent Indian consumers.
            - Historical behavioural features may be unavailable for
              a new applicant.
            - Linked-table chronology must be verified to ensure that
              only information available at the application date is used.

            **Model limitations**
            - Predicted probabilities are estimates, not guarantees.
            - SHAP describes the model's associations, not causal effects.
            - Model explanations are not financial advice.
            - Performance on a held-out dataset may differ from
              performance on future applicants.

            **Financial score limitations**
            - The score depends on the accuracy of user-entered figures.
            - Its weights and thresholds are documented assumptions,
              not universal standards.
            - It is not a credit score and does not predict default.
            - It should not be used alone to make financial decisions.

            **Privacy**
            - Do not enter bank passwords, account credentials or
              unnecessary identifying information.
            - This app does not intentionally save calculator inputs
              to a database.
            - Review security and privacy before public deployment.
            """
        )

        st.subheader("Research artifacts")

        artifact_paths = [
            ARTIFACTS_DIR / "model_metadata.json",
            RESULTS_DIR / "final_locked_test_results.csv",
            RESULTS_DIR / "bootstrap_confidence_intervals.csv",
            RESULTS_DIR / "delong_roc_auc_tests.csv",
            RESULTS_DIR / "feature_family_ablation.csv",
            RESULTS_DIR / "calibration_curves.png",
            RESULTS_DIR / "shap_global_beeswarm.png",
            RESULTS_DIR / "shap_global_importance.png",
        ]

        for path in artifact_paths:
            status = "Available" if path.exists() else "Not found"
            st.write(f"**{path.relative_to(BASE_DIR)}** — {status}")


# ------------------------------------------------------------
# 8. FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "FinResilience AI | Academic project prototype | "
    "Rule-based resilience scoring and research explainability"
)
