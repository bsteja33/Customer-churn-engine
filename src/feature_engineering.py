"""Feature engineering transformations for churn prediction."""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def col_map() -> dict:
    """Map snake_case field names to actual dataset column names."""
    return {
        "Gender": "Gender",
        "SeniorCitizen": "Senior Citizen",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "tenure": "Tenure in Months",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection Plan",
        "TechSupport": "Premium Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "Contract": "Contract",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method",
        "MonthlyCharges": "Monthly Charge",
        "TotalCharges": "Total Charges",
        "Married": "Married",
        "NumberOfDependents": "Number of Dependents",
        "NumberOfReferrals": "Number of Referrals",
        "SatisfactionScore": "Satisfaction Score",
        "InternetType": "Internet Type",
        "Offer": "Offer",
        "Age": "Age",
        "AvgMonthlyGBDownload": "Avg Monthly GB Download",
        "AvgMonthlyLongDistanceCharges": "Avg Monthly Long Distance Charges",
        "CLTV": "CLTV",
        "Under30": "Under 30",
        "UnlimitedData": "Unlimited Data",
        "StreamingMusic": "Streaming Music",
        "ReferredAFriend": "Referred a Friend",
        "TotalRefunds": "Total Refunds",
        "TotalExtraDataCharges": "Total Extra Data Charges",
        "TotalLongDistanceCharges": "Total Long Distance Charges",
        "TotalRevenue": "Total Revenue",
    }


# Categorical fields whose user-facing "no value" option is the literal
# string ``"None"``. At training time the dataset encodes missing values
# as actual nulls, which Polars' ``to_dummies`` expands into a
# ``<col>_null`` column. The frontend's dropdown uses ``"None"`` as a
# human-readable label for that no-value option, so the inference path
# must translate ``"None"`` (and ``""``) to actual null before encoding.
NULL_SENTINEL = "None"
NULL_SENTINEL_FIELDS = {"Offer", "Internet Type"}


# Binary fields (kept here for backwards-compatible tests that pin
# which API keys map to 0/1 columns). The model expects a single 0/1
# column per field — see ``engineer_features_inference``.
BINARY_FIELDS = {
    "Partner", "Dependents", "Phone Service", "Multiple Lines",
    "Internet Service", "Online Security", "Online Backup",
    "Device Protection Plan", "Premium Tech Support",
    "Streaming TV", "Streaming Movies", "Paperless Billing",
    "Married", "Under 30", "Unlimited Data", "Streaming Music",
    "Referred a Friend",
}


def _coerce_null_sentinels(values: pd.Series) -> pd.Series:
    """Map ``"None"`` and ``""`` to ``pd.NA`` for null-sentinel fields."""
    return values.replace({NULL_SENTINEL: pd.NA, "": pd.NA})


# Categorical encoding tables.
#
# The model was trained on a 50K-row frame where every category was
# represented, so ``pl.to_dummies`` emitted one column per known
# category. At inference time we typically get a single record (or a
# small batch) that only contains a subset of categories. Without
# explicit knowledge of the full vocabulary, the inference frame would
# be missing every "absent" dummy column, and the model would see it
# as zero (which is the wrong value for the *present* category).
#
# To guarantee train/inference parity, we hard-code the categories the
# model was fit on, derived from the model's ``feature_name_``:
#   * ``Contract``      → Month-to-Month / One Year / Two Year
#   * ``Gender``        → Female / Male
#   * ``Internet Type`` → Cable / DSL / Fiber Optic / null
#   * ``Offer``         → Offer A..E / null
#   * ``Payment Method`` → Bank Withdrawal / Credit Card / Mailed Check
#   * ``Senior Citizen`` → "0" / "1" (stringified, then one-hot)

CATEGORICAL_ENCODINGS: dict[str, dict[str, str]] = {
    "Contract": {
        "Month-to-Month": "Contract_Month_to_Month",
        "One Year": "Contract_One_Year",
        "Two Year": "Contract_Two_Year",
    },
    "Gender": {
        "Female": "Gender_Female",
        "Male": "Gender_Male",
    },
    "Internet Type": {
        "Cable": "Internet_Type_Cable",
        "DSL": "Internet_Type_DSL",
        "Fiber Optic": "Internet_Type_Fiber_Optic",
    },
    "Offer": {
        "Offer A": "Offer_Offer_A",
        "Offer B": "Offer_Offer_B",
        "Offer C": "Offer_Offer_C",
        "Offer D": "Offer_Offer_D",
        "Offer E": "Offer_Offer_E",
    },
    "Payment Method": {
        "Bank Withdrawal": "Payment_Method_Bank_Withdrawal",
        "Credit Card": "Payment_Method_Credit_Card",
        "Mailed Check": "Payment_Method_Mailed_Check",
    },
}

# Columns that represent a "no value / null" category for each
# null-bearing field (the dataset's null became a ``<col>_null`` column
# at training time, so we emit it explicitly here).
NULL_CATEGORICAL_COLUMNS: dict[str, str] = {
    "Offer": "Offer_null",
    "Internet Type": "Internet_Type_null",
}


def _one_hot_categorical(
    df: pd.DataFrame,
    col: str,
    encoding: dict[str, str],
    null_column: str | None = None,
) -> pd.DataFrame:
    """Emit one-hot columns for a single categorical field.

    Always emits *every* column in ``encoding`` (plus ``null_column`` if
    given), so the produced frame is independent of which rows are
    present. This is what makes a single-record inference produce the
    same column set as a 50K-row training frame.
    """
    out = pd.DataFrame(index=df.index)
    if col in df.columns:
        values = df[col].astype("object").where(df[col].notna(), None)
        for input_value, output_col in encoding.items():
            out[output_col] = (values == input_value).astype("uint8")
        if null_column is not None:
            out[null_column] = values.isna().astype("uint8")
    else:
        # Field absent from the input — emit all-zero placeholders.
        for output_col in encoding.values():
            out[output_col] = 0
        if null_column is not None:
            out[null_column] = 0
    return out


def engineer_features_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw input DataFrame into model-ready feature matrix.

    Applies column renaming, ``"None"`` → null normalization for
    null-sentinel fields, explicit one-hot encoding of every known
    categorical (using the training vocabulary so the produced frame
    is shape-compatible with the model regardless of which record
    values are present), and regex column sanitization.

    Designed for the inference path (API and CLI predict).
    """
    mapping = col_map()
    rename = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename)

    if "Total Charges" in df.columns:
        df["Total Charges"] = (
            df["Total Charges"]
            .replace(r"^\s*$", "0.0", regex=True)
            .astype(float)
        )

    if "Senior Citizen" in df.columns:
        # Cast to string so we can explicitly emit both Senior_Citizen_0
        # and Senior_Citizen_1 (matching the Polars stringification +
        # to_dummies step in train._clean_features / _encode_features).
        df["Senior Citizen"] = df["Senior Citizen"].astype("Int64").astype(str)
        df["Senior Citizen"] = df["Senior Citizen"].where(
            df["Senior Citizen"].isin(("0", "1")), "0"
        )

    # Translate the FE's "None" string to actual null for the columns
    # whose training data uses null as the "no value" encoding.
    for col in NULL_SENTINEL_FIELDS:
        if col in df.columns:
            df[col] = _coerce_null_sentinels(df[col])

    # Coerce numerics: NaN → 0 for filled-forward inference.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)

    # Build the explicit one-hot blocks for every known categorical.
    one_hot_blocks: list[pd.DataFrame] = []
    for col, encoding in CATEGORICAL_ENCODINGS.items():
        null_col = NULL_CATEGORICAL_COLUMNS.get(col)
        one_hot_blocks.append(_one_hot_categorical(df, col, encoding, null_col))

    # Senior Citizen: explicit 0/1 one-hot.
    sc_block = pd.DataFrame(index=df.index)
    if "Senior Citizen" in df.columns:
        sc = df["Senior Citizen"].astype(str)
        sc_block["Senior_Citizen_0"] = (sc == "0").astype("uint8")
        sc_block["Senior_Citizen_1"] = (sc == "1").astype("uint8")
    else:
        sc_block["Senior_Citizen_0"] = 0
        sc_block["Senior_Citizen_1"] = 0
    one_hot_blocks.append(sc_block)

    # Concatenate one-hot columns and drop the originals (we want the
    # encoded form, not the raw categorical).
    if one_hot_blocks:
        encoded = pd.concat(one_hot_blocks, axis=1)
        df = pd.concat([df, encoded], axis=1)
    for col in CATEGORICAL_ENCODINGS:
        if col in df.columns:
            df = df.drop(columns=[col])
    if "Senior Citizen" in df.columns:
        df = df.drop(columns=["Senior Citizen"])

    # Sanitize column names: drop any non-alphanumeric/underscore chars
    # (handles the few remaining "spaces" in numeric column names like
    # ``Phone Service`` → ``Phone_Service``).
    df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in df.columns]

    # Coerce any non-numeric leftovers to numeric so the LightGBM
    # pipeline (which expects a fully numeric frame) never sees a bool
    # or object dtype.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list]:
    """Build feature matrix, target vector, and column type lists."""
    df = df.copy()

    if "Total Revenue" in df.columns and "Tenure in Months" in df.columns:
        df["Revenue_per_Tenure"] = np.where(
            df["Tenure in Months"] == 0, 0,
            df["Total Revenue"] / df["Tenure in Months"]
        )

    if "Total Charges" in df.columns and "Tenure in Months" in df.columns:
        df["Charges_per_Month"] = np.where(
            df["Tenure in Months"] == 0, 0,
            df["Total Charges"] / df["Tenure in Months"]
        )

    service_cols = [
        col for col in df.columns
        if any(k in col for k in ["Security", "Backup", "Protection", "Tech Support", "Streaming"])
    ]
    if not service_cols:
        logger.warning("NO_SERVICE_COLUMNS_DETECTED")
    service_count = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    for svc in service_cols:
        if svc in df.columns:
            service_count += (df[svc] == "Yes").astype(int)
    df["Total_Services"] = service_count

    if "Age" in df.columns:
        df["Age_Group"] = (
            pd.cut(
                df["Age"],
                bins=[0, 30, 50, 70, 120],
                labels=["Young", "Adult", "Senior", "Elderly"],
            )
            .astype(str)
        )
        df = df.drop(columns=["Age"])

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    logger.info("FEATURE_ENGINEERING_COMPLETE: NUM=%d CAT=%d SAMPLES=%d", len(num_cols), len(cat_cols), len(X))
    return X, y, cat_cols, num_cols
