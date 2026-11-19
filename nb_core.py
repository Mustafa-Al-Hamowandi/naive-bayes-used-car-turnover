"""
nb_core.py - all calculations for the Naive Bayes project (no Streamlit code here).

Every function uses simple counting, exactly like the written report:
    Fast Sale = sold in 30 days or less,  Slow Sale = sold in more than 30 days.
"""
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

FAST_DAYS = 30
CLASSES = ["Fast", "Slow"]
FEATURES = ["Type", "Mileage_Band", "Condition"]
FEATURE_LABELS = {"Type": "Vehicle type", "Mileage_Band": "Mileage band", "Condition": "Condition", "Price_Level": "Price level"}
# Display order for feature values (values not in the data are still listed so
# students can try a Truck or a Hatchback and see "Cannot classify").
VALUE_ORDER = {
    "Type": ["Sedan", "SUV", "Coupe", "Truck", "Hatchback"],
    "Mileage_Band": ["Low", "Medium", "High", "Very High"],
    "Condition": ["Excellent", "Good", "Fair"],
    "Price_Level": ["Lower price", "High price"],  # only used to explore the independence assumption
}
# Features the Independence tab can compare (the model itself only uses FEATURES)
EXPLORE_FEATURES = FEATURES + ["Price_Level"]
DEFAULT_FILES = [
    "Spangled_Star_Dataset_Corrected__version_2_.xlsb",
    "Spangled_Star_Dataset.csv",
    "Spangled_Star_Dataset_Corrected__version_2_.xlsx",
]


# ----------------------------------------------------------------------------
# Loading and preparing the data
# ----------------------------------------------------------------------------
def read_any(source) -> pd.DataFrame:
    """Read .xlsb / .xlsx / .csv from a file path or an uploaded file object."""
    name = str(getattr(source, "name", source)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(source)
    if name.endswith(".xlsb"):
        return pd.read_excel(source, engine="pyxlsb")  # needs: pip install pyxlsb
    return pd.read_excel(source)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and add a Class column (Fast / Slow) for sold cars."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Condotion": "Condition"})  # the dataset spells it "Condotion"
    needed = ["Vehicle_ID", "Type", "Mileage_Band", "Condition", "Status", "Days_On_Lot"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"The file is missing these columns: {missing}")
    for col in ["Type", "Mileage_Band", "Condition", "Status"]:
        df[col] = df[col].astype(str).str.strip()
    df["Class"] = None
    sold_mask = df["Status"].str.lower() == "sold"
    df.loc[sold_mask, "Class"] = df.loc[sold_mask, "Days_On_Lot"].apply(
        lambda d: "Fast" if d <= FAST_DAYS else "Slow"
    )
    # Numeric mileage and price (used only for the independence examples, not by the model)
    if "Mileage" in df.columns:
        df["Mileage_Num"] = pd.to_numeric(df["Mileage"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "Listed_Price" in df.columns:
        price = pd.to_numeric(df["Listed_Price"].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False), errors="coerce")
        df["Price_Num"] = price
        median = price[sold_mask].median()  # median price of the sold cars
        df["Price_Level"] = np.where(price > median, "High price", "Lower price")
    return df


def price_threshold(sold: pd.DataFrame):
    """Median listed price of the sold cars (the line between 'Lower price' and 'High price')."""
    return float(sold["Price_Num"].median()) if "Price_Num" in sold.columns else None


def split(df: pd.DataFrame):
    """Return (sold, unsold) data frames."""
    sold = df[df["Status"].str.lower() == "sold"].copy()
    unsold = df[df["Status"].str.lower() == "available"].copy()
    return sold, unsold


def find_default_file(folder: Path):
    for name in DEFAULT_FILES:
        p = folder / name
        if p.exists():
            return p
    return None


# ----------------------------------------------------------------------------
# Counting helpers
# ----------------------------------------------------------------------------
def class_counts(sold: pd.DataFrame) -> dict:
    n_fast = int((sold["Class"] == "Fast").sum())
    n_slow = int((sold["Class"] == "Slow").sum())
    return {"Fast": n_fast, "Slow": n_slow, "Total": n_fast + n_slow}


def count_value(sold, feature, value, cls=None) -> int:
    """How many sold cars have feature == value (optionally only in class cls)."""
    mask = sold[feature] == value
    if cls is not None:
        mask &= sold["Class"] == cls
    return int(mask.sum())


def count_two(sold, f1, v1, f2, v2, cls) -> int:
    """How many cars of class cls have BOTH f1 == v1 and f2 == v2."""
    mask = (sold[f1] == v1) & (sold[f2] == v2) & (sold["Class"] == cls)
    return int(mask.sum())


# ----------------------------------------------------------------------------
# Section 3: conditional probability and Bayes' theorem
# ----------------------------------------------------------------------------
def bayes_terms(sold, feature, value, cls) -> dict:
    """All the pieces of Bayes' theorem for P(cls | feature = value)."""
    cc = class_counts(sold)
    N, n_c = cc["Total"], cc[cls]
    n_val = count_value(sold, feature, value)
    n_both = count_value(sold, feature, value, cls)
    prior = n_c / N
    likelihood = n_both / n_c if n_c else 0.0
    evidence = n_val / N
    posterior = (likelihood * prior / evidence) if evidence else None
    direct = (n_both / n_val) if n_val else None
    other = "Slow" if cls == "Fast" else "Fast"
    lik_other = count_value(sold, feature, value, other) / cc[other]
    evidence_sum = likelihood * prior + lik_other * (cc[other] / N)  # second way to get evidence
    return dict(N=N, n_class=n_c, n_value=n_val, n_both=n_both, prior=prior,
                likelihood=likelihood, evidence=evidence, posterior=posterior,
                direct=direct, other=other, lik_other=lik_other,
                prior_other=cc[other] / N, evidence_sum=evidence_sum)


# ----------------------------------------------------------------------------
# Section 4: Naive Bayes score and percentage for one car
# ----------------------------------------------------------------------------
def nb_scores(sold, values: dict) -> dict:
    """
    values: {"Type": "Sedan", "Mileage_Band": "Low", "Condition": "Excellent"}
    Returns, for each class, the factors (prior and likelihoods), the running
    product after each factor, the final score, the percentage, and the prediction.
    """
    cc = class_counts(sold)
    N = cc["Total"]
    out = {"values": dict(values), "N": N}
    exact = {}
    for cls in CLASSES:
        n_c = cc[cls]
        factors = [{"label": f"Prior P({cls})", "num": n_c, "den": N}]
        for f in FEATURES:
            n = count_value(sold, f, values[f], cls)
            factors.append({"label": f"P({values[f]} | {cls})", "num": n, "den": n_c})
        prod = Fraction(1)
        running = []
        for fac in factors:
            fac["value"] = fac["num"] / fac["den"] if fac["den"] else 0.0
            prod *= Fraction(fac["num"], fac["den"]) if fac["den"] else 0
            running.append(float(prod))
        exact[cls] = prod
        out[cls] = {"factors": factors, "running": running, "score": float(prod),
                    "top": _product([f["num"] for f in factors]),
                    "bottom": _product([f["den"] for f in factors])}
    total = exact["Fast"] + exact["Slow"]
    out["total"] = float(total)
    for cls in CLASSES:
        out[cls]["percent"] = float(exact[cls] / total * 100) if total else None
    if total == 0:
        out["prediction"] = "Cannot classify"
    elif exact["Fast"] > exact["Slow"]:
        out["prediction"] = "Fast"
    elif exact["Slow"] > exact["Fast"]:
        out["prediction"] = "Slow"
    else:
        out["prediction"] = "Tie"
    return out


def _product(nums):
    p = 1
    for n in nums:
        p *= n
    return p


# ----------------------------------------------------------------------------
# Section 6: predictions for all unsold cars
# ----------------------------------------------------------------------------
def predict_unsold(sold, unsold) -> pd.DataFrame:
    rows = []
    for _, r in unsold.iterrows():
        vals = {f: r[f] for f in FEATURES}
        res = nb_scores(sold, vals)
        rows.append({"Vehicle_ID": r["Vehicle_ID"], **vals,
                     "Fast score": res["Fast"]["score"], "Slow score": res["Slow"]["score"],
                     "Prediction": res["prediction"]})
    return pd.DataFrame(rows)


def group_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Group cars with identical features (Table A2 in the report)."""
    g = (pred.groupby(FEATURES + ["Fast score", "Slow score", "Prediction"], as_index=False)
             .size().rename(columns={"size": "Cars"}))
    order = {"Fast": 0, "Slow": 1, "Cannot classify": 2, "Tie": 3}
    g["_o"] = g["Prediction"].map(order)
    return g.sort_values(["_o", "Type", "Mileage_Band"]).drop(columns="_o").reset_index(drop=True)


def prediction_counts(pred: pd.DataFrame) -> dict:
    vc = pred["Prediction"].value_counts()
    return {k: int(vc.get(k, 0)) for k in ["Fast", "Slow", "Cannot classify"]}


# ----------------------------------------------------------------------------
# Section 6: which features matter (fast-sale rate per feature value)
# ----------------------------------------------------------------------------
def feature_rates(sold) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        for v in VALUE_ORDER[f]:
            n = count_value(sold, f, v)
            if n == 0:
                continue
            k = count_value(sold, f, v, "Fast")
            rows.append({"Feature": FEATURE_LABELS[f], "Value": v, "Fast": k, "Sold cars": n,
                         "Fast-sale rate (%)": k / n * 100})
    return pd.DataFrame(rows)


def fmt(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


# ----------------------------------------------------------------------------
# Section 5: where the independence assumption is NOT true (mileage and price)
# ----------------------------------------------------------------------------
def price_mileage_stats(df: pd.DataFrame, sold: pd.DataFrame):
    """
    Correlation r between mileage and price (all cars, Fast Sales only, Slow Sales only)
    and the average price in each mileage band.
    """
    def corr(x):
        x = x.dropna(subset=["Mileage_Num", "Price_Num"])
        return (float(x["Mileage_Num"].corr(x["Price_Num"])), len(x)) if len(x) > 2 else (None, len(x))

    r = {"all": corr(df), "Fast": corr(sold[sold["Class"] == "Fast"]), "Slow": corr(sold[sold["Class"] == "Slow"])}
    bands = (df.groupby("Mileage_Band")["Price_Num"].agg(["mean", "count"])
               .reindex(VALUE_ORDER["Mileage_Band"]).dropna())
    return r, bands