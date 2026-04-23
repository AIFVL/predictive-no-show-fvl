import numpy as np
import pandas as pd


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build additional features using only existing columns (no leakage).
    Returns a new DataFrame.
    """
    out = df.copy()

    # ---- Basic safety cleanup ----
    out.columns = [str(c).strip() for c in out.columns]
    for c in list(out.columns):
        if "Unnamed" in c:
            out.drop(columns=[c], inplace=True)

    # ---- Known column names ----
    col_age = "Age"
    col_diseases = "Number of Diseases"
    col_hosp = "Recent Hospitalization"
    col_meds = "Number of Medications"
    col_hour = "Hour"
    col_day = "Day"
    col_month = "Month"
    col_lead = "Creation to Assignment Interval"
    col_prev_att = "Number of Previous Attendance"
    col_prev_no = "Number of Previous Non-Attendance"

    # ---- Ratios and totals ----
    if col_prev_att in out.columns and col_prev_no in out.columns:
        prev_total = out[col_prev_att].fillna(0) + out[col_prev_no].fillna(0)
        out["Prev_Total"] = prev_total
        out["Has_Prev"] = (prev_total > 0).astype(int)
        out["Prev_NoShow_Rate"] = np.where(prev_total > 0, out[col_prev_no] / prev_total, 0.0)
        out["Prev_Show_Rate"] = np.where(prev_total > 0, out[col_prev_att] / prev_total, 0.0)

    # ---- Complexity / interaction features ----
    if col_diseases in out.columns and col_meds in out.columns:
        out["Disease_Med_Ratio"] = out[col_diseases] / (out[col_meds] + 1.0)
        out["Clinical_Burden"] = out[col_diseases] + out[col_meds]

    if col_hosp in out.columns and col_diseases in out.columns:
        out["Hosp_x_Diseases"] = out[col_hosp] * out[col_diseases]

    # ---- Log transform for skewed lead time ----
    if col_lead in out.columns:
        out["Lead_Time_Log"] = np.log1p(out[col_lead].clip(lower=0))

    # ---- Cyclical encodings ----
    if col_hour in out.columns:
        out["Hour_Sin"] = np.sin(2 * np.pi * out[col_hour] / 24.0)
        out["Hour_Cos"] = np.cos(2 * np.pi * out[col_hour] / 24.0)

    if col_day in out.columns:
        out["Day_Sin"] = np.sin(2 * np.pi * out[col_day] / 7.0)
        out["Day_Cos"] = np.cos(2 * np.pi * out[col_day] / 7.0)

    if col_month in out.columns:
        out["Month_Sin"] = np.sin(2 * np.pi * out[col_month] / 12.0)
        out["Month_Cos"] = np.cos(2 * np.pi * out[col_month] / 12.0)

    # ---- Bins (kept categorical for one-hot later) ----
    if col_age in out.columns:
        bins = [0, 17, 39, 59, 79, 120]
        labels = ["0_17", "18_39", "40_59", "60_79", "80_plus"]
        out["Age_Bin"] = pd.cut(out[col_age], bins=bins, labels=labels, include_lowest=True)

    if col_hour in out.columns:
        hour_bins = [-1, 11, 17, 23]
        hour_labels = ["Morning", "Afternoon", "Evening"]
        out["Hour_Bin"] = pd.cut(out[col_hour], bins=hour_bins, labels=hour_labels)

    return out

