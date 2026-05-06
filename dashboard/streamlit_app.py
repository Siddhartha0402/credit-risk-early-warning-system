"""Streamlit dashboard for credit risk predictions."""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_customer_risk


# FastAPI remains available in api/app.py for API deployment and local API
# workflows. Start it locally with:
# .venv\Scripts\python.exe -m uvicorn api.app:app --reload

# Streamlit uses direct model inference so the deployed dashboard can run as a
# single service without requiring the FastAPI process to be running.


st.set_page_config(
    page_title="Credit Risk Early Warning Dashboard",
    layout="wide",
)


st.title("Credit Risk Scoring and Early Default Warning System")
st.write(
    "This dashboard scores a customer application using the trained model "
    "artifacts. It estimates the probability of default, assigns a low, "
    "medium, or high risk band, and returns the recommended next action."
)

st.info(
    "Risk interpretation: Low Risk is below 30%, Medium Risk is between 30% "
    "and 60%, and High Risk is above 60% default probability."
)


with st.sidebar:
    st.header("Customer Inputs")

    with st.form("risk_prediction_form"):
        amt_income_total = st.number_input(
            "Income",
            min_value=0.0,
            value=157500.0,
            step=1000.0,
            help="Total annual customer income.",
        )
        amt_credit = st.number_input(
            "Credit Amount",
            min_value=0.0,
            value=497520.0,
            step=1000.0,
            help="Total requested credit amount.",
        )
        amt_annuity = st.number_input(
            "Loan Annuity",
            min_value=0.0,
            value=27000.0,
            step=500.0,
            help="Recurring loan annuity amount.",
        )
        amt_goods_price = st.number_input(
            "Goods Price",
            min_value=0.0,
            value=450000.0,
            step=1000.0,
            help="Price of goods for the loan application.",
        )
        days_birth = st.number_input(
            "Age in Days",
            value=-14500,
            step=30,
            help="Use negative days before today, for example -14500 is about 40 years old.",
        )
        days_employed = st.number_input(
            "Employment Days",
            value=-2400,
            step=30,
            help="Use negative days before today, for example -2400 is about 6.5 years employed.",
        )
        cnt_children = st.number_input(
            "CNT_CHILDREN",
            min_value=0,
            value=1,
            step=1,
            help="Number of children.",
        )
        cnt_fam_members = st.number_input(
            "CNT_FAM_MEMBERS",
            min_value=1.0,
            value=3.0,
            step=1.0,
            help="Total family member count.",
        )
        name_contract_type = st.selectbox(
            "NAME_CONTRACT_TYPE",
            options=["Cash loans", "Revolving loans"],
        )
        code_gender = st.selectbox(
            "CODE_GENDER",
            options=["F", "M"],
        )
        flag_own_car = st.selectbox(
            "FLAG_OWN_CAR",
            options=["N", "Y"],
        )
        flag_own_realty = st.selectbox(
            "FLAG_OWN_REALTY",
            options=["Y", "N"],
        )

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)


payload = {
    "AMT_INCOME_TOTAL": float(amt_income_total),
    "AMT_CREDIT": float(amt_credit),
    "AMT_ANNUITY": float(amt_annuity),
    "AMT_GOODS_PRICE": float(amt_goods_price),
    "DAYS_BIRTH": int(days_birth),
    "DAYS_EMPLOYED": int(days_employed),
    "CNT_CHILDREN": int(cnt_children),
    "CNT_FAM_MEMBERS": float(cnt_fam_members),
    "NAME_CONTRACT_TYPE": name_contract_type,
    "CODE_GENDER": code_gender,
    "FLAG_OWN_CAR": flag_own_car,
    "FLAG_OWN_REALTY": flag_own_realty,
}


left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Application Snapshot")
    st.dataframe(
        pd.DataFrame([payload]).T.rename(columns={0: "Value"}),
        use_container_width=True,
    )

with right_col:
    st.subheader("Prediction Results")

    if not submitted:
        st.caption("Enter customer details and click Predict Risk to score the application.")
    else:
        try:
            result = predict_customer_risk(payload)

            default_probability = float(result["default_probability"])
            probability_display = f"{default_probability:.1%}"
            risk_band = result["risk_band"]
            recommended_action = result["recommended_action"]

            metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
            metric_col_1.metric("Default Probability", probability_display)
            metric_col_2.metric("Risk Band", risk_band)
            metric_col_3.metric("Recommended Action", recommended_action)

            if risk_band == "Low Risk":
                st.success("Low Risk: default probability is below 30%.")
            elif risk_band == "Medium Risk":
                st.warning("Medium Risk: default probability is between 30% and 60%.")
            else:
                st.error("High Risk: default probability is above 60%.")

        except FileNotFoundError as exc:
            st.error(
                "Model artifact files are missing. Confirm that "
                "models/credit_risk_model.pkl and models/feature_columns.pkl "
                f"exist before deploying this Streamlit app. Details: {exc}"
            )
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"Unexpected prediction response format: {exc}")
