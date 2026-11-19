"""
Streamlit app: Predicting used-car inventory turnover with Naive Bayes.
Every tab shows one calculation from the written report, with the numbers and a chart.

Run:   streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import nb_core as nb
import plots as pl

st.set_page_config(page_title="Naive Bayes: Used-Car Sales", page_icon="🚗", layout="wide")

# Make sure all three code files are the same version (app.py, nb_core.py, plots.py)
if not hasattr(nb, "EXPLORE_FEATURES") or not hasattr(pl, "mileage_price_scatter"):
    st.error("nb_core.py or plots.py is an old version. Replace all three files (app.py, nb_core.py, plots.py) "
             "with the new ones, then stop the app (Ctrl+C) and run it again.")
    st.stop()

CLASS_NAME = {"Fast": "Fast Sale", "Slow": "Slow Sale"}

# Colors (same as the charts)
NAVY, TEAL, RED, AMBER, BLUE, GRAY = "#1F3A5F", "#2A7F9E", "#C8553D", "#E0A030", "#3B82A0", "#9CA3AF"


def show(fig):
    """Show a matplotlib figure and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


def frac(num, den):
    return f"{num}/{den}"


def card(col, label, value, sub="", color=TEAL):
    """A colored number card."""
    col.markdown(
        f'<div class="card" style="--c:{color}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Look and feel (colors, cards, tabs, sidebar)
# ----------------------------------------------------------------------------
CSS = """
<style>
.stApp {background: linear-gradient(180deg,#E6F0F7 0%,#F5F9FC 35%,#F4F8FB 100%);}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.4rem; max-width: 1280px;}
footer, #MainMenu {visibility: hidden;}
[data-testid="stAppDeployButton"], .stDeployButton {display: none;}

/* sidebar */
[data-testid="stSidebar"] {background: linear-gradient(180deg,#1F3A5F 0%,#24506E 55%,#2A7F9E 100%);}
[data-testid="stSidebar"] * {color: #F3F8FB !important;}
.side-title {font-size: 1.35rem; font-weight: 800; margin-bottom: 2px;}
.side-sub {opacity: .85; margin-bottom: 14px;}
.side-box {background: rgba(255,255,255,.12); border-radius: 12px; padding: 10px 14px; margin: 10px 0; font-size: .9rem;}
.chip {display:inline-block; padding:3px 12px; border-radius:999px; font-weight:700; font-size:.8rem; margin:2px 4px 2px 0;}
.chip.fast {background:#2A7F9E; color:#fff !important; border:1px solid rgba(255,255,255,.5);}
.chip.slow {background:#C8553D; color:#fff !important; border:1px solid rgba(255,255,255,.5);}

/* hero banner */
.hero {background: linear-gradient(120deg,#1F3A5F 0%,#2A7F9E 100%); border-radius: 18px; padding: 26px 32px; margin-bottom: 18px;
       box-shadow: 0 6px 20px rgba(31,58,95,.25);}
.hero h1 {color:#fff !important; font-size: 2.1rem; margin: 0 0 4px 0; padding: 0; border: none;}
.hero p {color: #DCEBF3; margin: 0 0 12px 0; font-size: 1.02rem;}
.hero .pill {display:inline-block; padding:4px 14px; border-radius:999px; font-weight:700; font-size:.82rem; margin-right:8px; color:#fff;}
.hero .pill.fast {background: rgba(255,255,255,.18); border:1px solid #8FD3E8;}
.hero .pill.slow {background: rgba(255,255,255,.18); border:1px solid #F4A896;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: 6px; background: #fff; padding: 6px; border-radius: 14px; flex-wrap: wrap;
       box-shadow: 0 2px 10px rgba(31,58,95,.10);}
.stTabs [data-baseweb="tab"] {height: 42px; padding: 0 16px; border-radius: 10px; background: transparent;}
.stTabs [data-baseweb="tab"] p {color: #1F3A5F; font-weight: 600; font-size: .95rem;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg,#1F3A5F,#2A7F9E);}
.stTabs [aria-selected="true"] p {color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

/* headings */
.block-container h2 {color: #1F3A5F; border-left: 7px solid #2A7F9E; padding-left: 12px; margin-top: .6rem;}
.block-container h3 {color: #2A7F9E;}

/* cards */
.card {background:#fff; border-radius:14px; padding:14px 18px; border-top:5px solid var(--c);
       box-shadow:0 2px 10px rgba(31,58,95,.12); height:100%;}
.card .label {font-size:.78rem; color:#5B6B7B; text-transform:uppercase; letter-spacing:.05em; font-weight:700;}
.card .value {font-size:2.1rem; font-weight:800; color:var(--c); line-height:1.15;}
.card .sub {font-size:.85rem; color:#5B6B7B;}

/* tables, charts, formulas, messages */
[data-testid="stDataFrame"] {border-radius:12px; overflow:hidden; box-shadow:0 2px 10px rgba(31,58,95,.10);}
[data-testid="stImage"] img {border-radius:12px; background:#fff; box-shadow:0 2px 10px rgba(31,58,95,.10);}
[data-testid="stLatex"] {background:#fff; border-radius:12px; padding:8px 12px; box-shadow:0 2px 10px rgba(31,58,95,.08);}
[data-testid="stAlert"] {border-radius:12px;}
[data-testid="stExpander"] {background:#fff; border-radius:12px; border:1px solid #D3E3ED;}
[data-testid="stException"] a[href*="google.com"], [data-testid="stException"] a[href*="chatgpt.com"],
[data-testid="stException"] a[href*="openai.com"] {display: none !important;}
[data-testid="stFileUploader"] section {background:#fff; border:2px dashed #2A7F9E; border-radius:14px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
  <h1>🚗 Predicting Used-Car Inventory Turnover</h1>
  <p>Conditional probability, Bayes' theorem, and Naïve Bayes classification &nbsp;|&nbsp; Spangled Star Auto Sales</p>
  <span class="pill fast">Fast Sale = sold in 30 days or less</span>
  <span class="pill slow">Slow Sale = sold in more than 30 days</span>
</div>
""",
    unsafe_allow_html=True,
)

notice = st.container()  # messages about the data appear here, above the tabs

tabs = st.tabs(["📊 Dataset", "🎯 Conditional probability", "🔄 Bayes' theorem", "🚗 Naïve Bayes example",
                "🔗 Independence", "📈 Results", "⭐ Which features matter", "📁 Upload data"])

# ----------------------------------------------------------------------------
# Upload tab (the uploader lives here so it does not take space on the main screen)
# ----------------------------------------------------------------------------
with tabs[7]:
    st.header("Upload data")
    st.write("The app uses the dataset that is saved next to `app.py`. "
             "To use a different file, upload it here (.xlsb, .xlsx or .csv). "
             "Remove the uploaded file with the ✕ button to go back to the saved dataset.")
    uploaded = st.file_uploader("Choose a dataset file", type=["xlsb", "xlsx", "csv"])


@st.cache_data
def load_path(path_str):
    return nb.prepare(nb.read_any(path_str))


try:
    if uploaded is not None:
        df = nb.prepare(nb.read_any(uploaded))
        source_name = uploaded.name + " (uploaded)"
    else:
        default = nb.find_default_file(Path(__file__).parent)
        if default is None:
            with notice:
                st.warning("No dataset found. Open the **📁 Upload data** tab and upload the file, "
                           "or save it in the same folder as app.py.")
            st.stop()
        df = load_path(str(default))
        source_name = default.name
except Exception as e:  # show a friendly message if the file cannot be read
    with notice:
        st.error(f"Could not read the dataset: {e}")
    st.stop()

sold, unsold = nb.split(df)
counts = nb.class_counts(sold)
N = counts["Total"]

# Data status (upload tab) and sidebar summary
with tabs[7]:
    st.success(f"Using: **{source_name}**")
    need = ["Vehicle_ID", "Type", "Mileage_Band", "Condition", "Status", "Days_On_Lot"]
    st.markdown("**Data check** (columns the app needs)")
    st.dataframe(pd.DataFrame({"Column": need, "Found": ["✅" if c in df.columns else "❌" for c in need]}), hide_index=True)
    k = st.columns(3)
    card(k[0], "Rows in file", len(df), "", NAVY)
    card(k[1], "Sold cars", N, "used to build the model", TEAL)
    card(k[2], "Unsold cars", len(unsold), "used for prediction", AMBER)

st.sidebar.markdown(
    f"""
<div class="side-title">🚗 Naïve Bayes Project</div>
<div class="side-sub">Math for Data Science &amp; AI</div>
<div class="side-box"><b>Dataset</b><br>{source_name}<br>{len(df)} cars: {N} sold, {len(unsold)} unsold</div>
<div class="side-box"><b>Classes</b><br>
<span class="chip fast">Fast ≤ 30 days</span><span class="chip slow">Slow &gt; 30 days</span></div>
<div class="side-box"><b>How to use</b><br>Open a tab and choose a feature, a value, or a car. The numbers and charts update right away.</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helper: choose "class + feature value" (used in tabs 2 and 3)
# ----------------------------------------------------------------------------
def pick_event(key):
    c1, c2, c3 = st.columns(3)
    feat = c1.selectbox("Feature", nb.FEATURES, format_func=lambda f: nb.FEATURE_LABELS[f], key=f"{key}_f")
    values = [v for v in nb.VALUE_ORDER[feat] if nb.count_value(sold, feat, v) > 0]
    val = c2.selectbox("Value", values, key=f"{key}_v_{feat}")
    cls = c3.radio("Class", nb.CLASSES, horizontal=True, format_func=lambda c: CLASS_NAME[c], key=f"{key}_c")
    return feat, val, cls


# ----------------------------------------------------------------------------
# Helper: full Naive Bayes calculation for one car (used in tabs 4 and 6)
# ----------------------------------------------------------------------------
def render_nb(res):
    vals = res["values"]
    st.markdown(f"**Car:** {vals['Type']}, {vals['Mileage_Band']} mileage, {vals['Condition']} condition")

    # Step 0: the numbers we look up
    st.subheader("Step 1: Look up the numbers")
    rows = []
    for i in range(4):
        f, s = res["Fast"]["factors"][i], res["Slow"]["factors"][i]
        label = "Prior" if i == 0 else f["label"].split(" |")[0] + " | C)"
        rows.append({"Number": label,
                     "Fast Sale": f"{frac(f['num'], f['den'])} = {f['value']:.3f}",
                     "Slow Sale": f"{frac(s['num'], s['den'])} = {s['value']:.3f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    # Step 2: the score (multiply)
    st.subheader("Step 2: Multiply to get the score")
    st.latex(r"\text{Score} = P(C)\times P(x_1\mid C)\times P(x_2\mid C)\times P(x_3\mid C)")
    names = ["1. Prior", "2. × P(type | C)", "3. × P(mileage | C)", "4. × P(condition | C)"]
    steps = []
    for i, name in enumerate(names):
        row = {"Step": name}
        for cls in nb.CLASSES:
            fac = res[cls]["factors"][i]
            run = res[cls]["running"]
            if i == 0:
                row[CLASS_NAME[cls]] = f"{frac(fac['num'], fac['den'])} = {run[0]:.5f}"
            else:
                row[CLASS_NAME[cls]] = f"{run[i - 1]:.5f} × {frac(fac['num'], fac['den'])} = {run[i]:.5f}"
        steps.append(row)
    steps.append({"Step": "Score (rounded)", "Fast Sale": f"{res['Fast']['score']:.4f}", "Slow Sale": f"{res['Slow']['score']:.4f}"})
    st.dataframe(pd.DataFrame(steps), hide_index=True)

    with st.expander("One-step check (multiply all top numbers and all bottom numbers)"):
        for cls in nb.CLASSES:
            f = res[cls]["factors"]
            top = " × ".join(str(x["num"]) for x in f)
            bottom = " × ".join(str(x["den"]) for x in f)
            st.write(f"**{CLASS_NAME[cls]}:** ({top}) ÷ ({bottom}) = {res[cls]['top']:,} ÷ {res[cls]['bottom']:,} = {res[cls]['score']:.4f}")

    # Step 3: percentage
    st.subheader("Step 3: Turn the scores into percentages")
    st.latex(r"\text{Percentage} = \frac{\text{Score}}{\text{Fast score} + \text{Slow score}}")
    if res["total"] == 0:
        st.warning("Both scores are 0, so the total is 0 and we cannot divide. This car **cannot be classified**.")
    else:
        rf, rs = round(res["Fast"]["score"], 5), round(res["Slow"]["score"], 5)
        tot = round(rf + rs, 5)  # add the rounded scores, so every line can be checked by hand
        if tot == 0:  # very small scores: fall back to the exact values
            rf, rs, tot = res["Fast"]["score"], res["Slow"]["score"], res["total"]
        pct = pd.DataFrame([
            {"Step": "1. Score (5 decimals)", "Fast Sale": f"{rf:.5f}", "Slow Sale": f"{rs:.5f}"},
            {"Step": "2. Total of both scores", "Fast Sale": f"{rf:.5f} + {rs:.5f} = {tot:.5f}", "Slow Sale": f"{tot:.5f}"},
            {"Step": "3. Score ÷ total", "Fast Sale": f"{rf:.5f} ÷ {tot:.5f} = {rf / tot:.4f}",
             "Slow Sale": f"{rs:.5f} ÷ {tot:.5f} = {rs / tot:.4f}"},
            {"Step": "4. Percentage", "Fast Sale": f"{rf / tot * 100:.1f}%", "Slow Sale": f"{rs / tot * 100:.1f}%"},
        ])
        st.dataframe(pct, hide_index=True)
        st.caption("We use 5 decimals in this table so you can check each line by hand. A calculator that keeps all decimals may differ in the last digit.")

    # Prediction
    if res["prediction"] in ("Fast", "Slow"):
        st.success(f"Prediction: **{CLASS_NAME[res['prediction']]}** (the bigger score wins)")
    elif res["prediction"] == "Cannot classify":
        st.warning("Prediction: **Cannot classify**. A score is a product, and one number in it is 0 "
                   "(this car type never appears in the sold cars), so both scores are 0.")
    else:
        st.info("The two scores are equal, so there is a tie.")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        show(pl.factor_chart(res))
    with c2:
        show(pl.running_product(res))
    if res["total"] > 0:
        show(pl.percent_bar(res))



def show_counted(cls, f1, v1, f2, v2):
    """Show the cars behind a 'real data' count, so students can see where the number comes from."""
    n_c = counts[cls]
    sub = sold[(sold["Class"] == cls) & (sold[f1] == v1)].copy()
    n12 = int(((sold["Class"] == cls) & (sold[f1] == v1) & (sold[f2] == v2)).sum())
    with st.expander("See the cars that were counted"):
        st.write(f"**Step 1:** take the {n_c} {CLASS_NAME[cls]} cars and keep the ones with **{v1}**. There are **{len(sub)}** of them:")
        if sub.empty:
            st.write("No cars match, so the count is 0.")
            return
        sub[f"Also {v2}?"] = (sub[f2] == v2).map({True: "✅ yes", False: "no"})
        cols = ["Vehicle_ID", f1, f2]
        for extra in ("Listed_Price", "Days_On_Lot"):
            if extra in sub.columns and extra not in cols:
                cols.append(extra)
        cols.append(f"Also {v2}?")
        st.dataframe(sub[cols], hide_index=True)
        st.write(f"**Step 2:** count the cars that also have **{v2}** (the ✅ rows). That is **{n12}**.")
        st.write(f"**Step 3:** divide by the number of {CLASS_NAME[cls]} cars: {n12} ÷ {n_c} = **{n12 / n_c:.4f}**.")


# ============================================================================
# TAB 1: Dataset
# ============================================================================
with tabs[0]:
    st.header("1. The dataset")
    m = st.columns(5)
    card(m[0], "Total cars", len(df), "in the dataset", NAVY)
    card(m[1], "Sold", N, "build the model", BLUE)
    card(m[2], "Unsold", len(unsold), "we predict these", AMBER)
    card(m[3], "Fast Sales", counts["Fast"], f"{counts['Fast'] / N * 100:.1f}% of sold", TEAL)
    card(m[4], "Slow Sales", counts["Slow"], f"{counts['Slow'] / N * 100:.1f}% of sold", RED)
    st.write("")
    c1, c2 = st.columns([1, 1])
    with c1:
        show(pl.class_bar(counts))
    with c2:
        st.markdown("**Prior probabilities** (the Fast and Slow rates before we look at any feature)")
        st.latex(rf"P(\text{{Fast}}) = \frac{{{counts['Fast']}}}{{{N}}} = {counts['Fast'] / N:.4f} = {counts['Fast'] / N * 100:.1f}\%")
        st.latex(rf"P(\text{{Slow}}) = \frac{{{counts['Slow']}}}{{{N}}} = {counts['Slow'] / N:.4f} = {counts['Slow'] / N * 100:.1f}\%")
        st.write(f"A sold car is a **Fast Sale** if Days_On_Lot ≤ {nb.FAST_DAYS}. "
                 f"That gives {counts['Fast']} Fast Sales. The other {N} − {counts['Fast']} = {counts['Slow']} are Slow Sales.")
    with st.expander("Look at the data"):
        st.dataframe(df.drop(columns=[c for c in ["Class", "Mileage_Num", "Price_Num", "Price_Level"] if c in df.columns]), hide_index=True)

# ============================================================================
# TAB 2: Conditional probability
# ============================================================================
with tabs[1]:
    st.header("2. Conditional probability")
    st.write("Conditional probability is the chance of event A when we already know that event B happened. "
             "We read P(A | B) as “the probability of A given B.”")
    st.latex(r"P(A \mid B) = \frac{P(A \cap B)}{P(B)}")
    feat, val, cls = pick_event("cp")
    t = nb.bayes_terms(sold, feat, val, cls)
    st.write(f"**A** = the car is a {CLASS_NAME[cls]}.  **B** = the car's {nb.FEATURE_LABELS[feat].lower()} is {val}.")
    table = pd.DataFrame([
        {"Quantity": f"P(A ∩ B): {CLASS_NAME[cls]} AND {val}", "Calculation": f"{t['n_both']} ÷ {N}", "Value": f"{t['n_both'] / N:.4f}"},
        {"Quantity": f"P(B): {val}", "Calculation": f"{t['n_value']} ÷ {N}", "Value": f"{t['evidence']:.4f}"},
        {"Quantity": f"P(A | B) = P(A ∩ B) ÷ P(B)", "Calculation": f"{t['n_both'] / N:.4f} ÷ {t['evidence']:.4f}",
         "Value": nb.fmt(t["direct"], 4)},
    ])
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(table, hide_index=True)
        if t["n_value"]:
            st.success(f"Shortcut by counting: {t['n_both']} of the {t['n_value']} sold cars with {val} are a {CLASS_NAME[cls]}. "
                       f"{t['n_both']} ÷ {t['n_value']} = **{t['direct']:.3f}** = {t['direct'] * 100:.1f}%")
    with c2:
        show(pl.waffle(sold, feat, val, f"Dark colors = {val}"))

# ============================================================================
# TAB 3: Bayes' theorem
# ============================================================================
with tabs[2]:
    st.header("3. Bayes' theorem")
    st.write("Bayes' theorem calculates the conditional probability of an event using prior knowledge and new evidence.")
    st.latex(r"P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}")
    feat, val, cls = pick_event("bt")
    t = nb.bayes_terms(sold, feat, val, cls)
    terms = pd.DataFrame([
        {"Term": "Prior, P(A)", "Meaning": f"Chance of {CLASS_NAME[cls]} before we look at the car",
         "Calculation": f"{t['n_class']} ÷ {N}", "Value": f"{t['prior']:.4f}"},
        {"Term": "Likelihood, P(B | A)", "Meaning": f"How common {val} is inside the {CLASS_NAME[cls]} cars",
         "Calculation": f"{t['n_both']} ÷ {t['n_class']}", "Value": f"{t['likelihood']:.4f}"},
        {"Term": "Evidence, P(B)", "Meaning": f"How common {val} is in all sold cars",
         "Calculation": f"{t['n_value']} ÷ {N}", "Value": f"{t['evidence']:.4f}"},
        {"Term": "Posterior, P(A | B)", "Meaning": f"New chance of {CLASS_NAME[cls]} after we see {val}",
         "Calculation": f"({t['likelihood']:.4f} × {t['prior']:.4f}) ÷ {t['evidence']:.4f}", "Value": nb.fmt(t["posterior"], 4)},
    ])
    st.dataframe(terms, hide_index=True)
    if t["posterior"] is not None:
        st.success(f"A car with {val} has a **{t['posterior'] * 100:.1f}%** chance of being a {CLASS_NAME[cls]} "
                   f"(the prior was {t['prior'] * 100:.1f}%). "
                   f"Check by counting: {t['n_both']} ÷ {t['n_value']} = {t['direct']:.4f}.")
    c1, c2 = st.columns(2)
    with c1:
        if t["posterior"] is not None:
            show(pl.prior_vs_posterior(t["prior"], t["posterior"], cls, val))
    with c2:
        st.markdown("**A second way to find the evidence**")
        st.write(f"A car with {val} is either a Fast Sale or a Slow Sale, so we add the two cases:")
        fast_part = (t["likelihood"] * t["prior"]) if cls == "Fast" else (t["lik_other"] * t["prior_other"])
        slow_part = (t["likelihood"] * t["prior"]) if cls == "Slow" else (t["lik_other"] * t["prior_other"])
        st.latex(rf"P(B) = P(B\mid \text{{Fast}})P(\text{{Fast}}) + P(B\mid \text{{Slow}})P(\text{{Slow}}) = {fast_part:.3f} + {slow_part:.3f} = {fast_part + slow_part:.3f}")
        show(pl.evidence_two_ways(t["evidence"], t["evidence_sum"], (fast_part, slow_part), val))

# ============================================================================
# TAB 4: Naive Bayes example
# ============================================================================
with tabs[3]:
    st.header("4. Naïve Bayes: classify one car")
    st.write("Pick the features of a car. The app multiplies the prior and the likelihoods to get a Fast score and a Slow score. "
             "The report example is a **Sedan** with **Low** mileage in **Excellent** condition.")
    st.latex(r"P(C \mid x_1,\ldots,x_n) \propto P(C)\times P(x_1\mid C)\times P(x_2\mid C)\times\cdots\times P(x_n\mid C)")
    c1, c2, c3 = st.columns(3)
    car = {
        "Type": c1.selectbox("Vehicle type", nb.VALUE_ORDER["Type"], key="nb_type"),
        "Mileage_Band": c2.selectbox("Mileage band", nb.VALUE_ORDER["Mileage_Band"], key="nb_mil"),
        "Condition": c3.selectbox("Condition", nb.VALUE_ORDER["Condition"], key="nb_cond"),
    }
    render_nb(nb.nb_scores(sold, car))

# ============================================================================
# TAB 5: Independence
# ============================================================================
with tabs[4]:
    st.header("5. The independence assumption")
    st.write("Naïve Bayes assumes that, once we know the class, the features do not affect each other. "
             "We can check this with the data: if two features are independent, the real joint probability "
             "should be close to the product of the two single probabilities.")
    st.latex(r"P(x_1, x_2 \mid C) = P(x_1 \mid C) \times P(x_2 \mid C)")

    has_price = "Price_Level" in sold.columns
    explore = [f for f in nb.EXPLORE_FEATURES if f in sold.columns]
    thr = nb.price_threshold(sold) if has_price else None

    st.subheader("Check any two features")
    if has_price:
        st.caption(f"Price level: **High price** = listed price above the median of the sold cars (${thr:,.0f}). "
                   "Price is not used by the model. It is here to show features that are connected.")
    cls = st.radio("Class", nb.CLASSES, horizontal=True, format_func=lambda c: CLASS_NAME[c], key="ind_c")
    c1, c2 = st.columns(2)
    f1 = c1.selectbox("First feature", explore, format_func=lambda f: nb.FEATURE_LABELS[f], key="ind_f1")
    v1 = c1.selectbox("Value", [v for v in nb.VALUE_ORDER[f1] if nb.count_value(sold, f1, v) > 0], key=f"ind_v1_{f1}")
    others = [f for f in explore if f != f1]
    f2 = c2.selectbox("Second feature", others, format_func=lambda f: nb.FEATURE_LABELS[f], key=f"ind_f2_{f1}")
    v2 = c2.selectbox("Value ", [v for v in nb.VALUE_ORDER[f2] if nb.count_value(sold, f2, v) > 0], key=f"ind_v2_{f1}_{f2}")
    n_c = counts[cls]
    n1, n2 = nb.count_value(sold, f1, v1, cls), nb.count_value(sold, f2, v2, cls)
    n12 = nb.count_two(sold, f1, v1, f2, v2, cls)
    p1, p2, actual = n1 / n_c, n2 / n_c, n12 / n_c
    st.dataframe(pd.DataFrame([
        {"Quantity": f"P({v1} | {cls})", "Calculation": f"{n1} ÷ {n_c}", "Value": f"{p1:.4f}"},
        {"Quantity": f"P({v2} | {cls})", "Calculation": f"{n2} ÷ {n_c}", "Value": f"{p2:.4f}"},
        {"Quantity": "Naive product", "Calculation": f"{p1:.4f} × {p2:.4f}", "Value": f"{p1 * p2:.4f}"},
        {"Quantity": f"Real data: P({v1} and {v2} | {cls})", "Calculation": f"{n12} ÷ {n_c}", "Value": f"{actual:.4f}"},
    ]), hide_index=True)
    show_counted(cls, f1, v1, f2, v2)
    ca, cb = st.columns(2)
    with ca:
        show(pl.joint_vs_product(actual, p1 * p2, f"{actual:.3f}", f"{p1 * p2:.3f}"))
    with cb:
        diff = abs(actual - p1 * p2)
        st.write(f"The difference is **{diff:.3f}**.")
        st.write("If the two bars are close, the independence assumption works well for these two features. "
                 "If they are far apart, the features are connected. With only 28 or 35 cars in each class, "
                 "small differences can happen by chance, so we do not draw a strong conclusion.")
        if has_price:
            st.info("Try it: choose **Mileage band = Low** and **Price level = High price**. "
                    "The bars are far apart, so these two features are connected. The example below explains why.")

    # ---------------------------------------------------------------- not always true
    if has_price and "Mileage_Num" in df.columns:
        st.divider()
        st.subheader("When the assumption is NOT true: mileage and price")
        st.write("Naïve Bayes multiplies the features as if each one is separate evidence. In real life, features are often "
                 "connected. In our data, **cars with more miles usually have a lower price**, and cars with fewer miles "
                 "usually have a higher price.")
        stats, bands = nb.price_mileage_stats(df, sold)
        k = st.columns(3)
        card(k[0], "All 100 cars", f"r = {stats['all'][0]:.2f}", f"mileage and price, n = {stats['all'][1]}", NAVY)
        card(k[1], "Fast Sales only", f"r = {stats['Fast'][0]:.2f}", f"same class, n = {stats['Fast'][1]}", TEAL)
        card(k[2], "Slow Sales only", f"r = {stats['Slow'][0]:.2f}", f"same class, n = {stats['Slow'][1]}", RED)
        st.caption("r is the correlation. It goes from −1 to +1. A negative r means: when mileage goes up, price goes down. "
                   "0 means no connection. Naïve Bayes needs the features to be independent **inside each class**, "
                   "so the Fast-only and Slow-only values matter most. Both are clearly negative.")
        g1, g2 = st.columns([6, 5])
        with g1:
            show(pl.mileage_price_scatter(df))
        with g2:
            show(pl.price_by_band(bands))
        low, high = bands.loc["Low", "mean"], bands.loc["High", "mean"]
        st.success(f"Low mileage cars have an average price of **${low:,.0f}**. High mileage cars have an average price of "
                   f"**${high:,.0f}**. If mileage and price were independent, the average price would be about the same in every band.")

# ============================================================================
# TAB 6: Results
# ============================================================================
with tabs[5]:
    st.header("6. Results for the unsold cars")
    pred = nb.predict_unsold(sold, unsold)
    pc = nb.prediction_counts(pred)
    U = len(pred)
    m = st.columns(3)
    card(m[0], "Predicted Fast", pc["Fast"], f"{pc['Fast'] / U * 100:.1f}% of {U} unsold cars", TEAL)
    card(m[1], "Predicted Slow", pc["Slow"], f"{pc['Slow'] / U * 100:.1f}% of {U} unsold cars", RED)
    card(m[2], "Cannot classify", pc["Cannot classify"], f"{pc['Cannot classify'] / U * 100:.1f}% of {U} unsold cars", GRAY)

    st.subheader("How we get the numbers")
    st.markdown(
        f"""
**Historical rows:** {counts['Fast']} ÷ {N} = {counts['Fast'] / N:.4f} = {counts['Fast'] / N * 100:.1f}% Fast, and
{counts['Slow']} ÷ {N} = {counts['Slow'] / N:.4f} = {counts['Slow'] / N * 100:.1f}% Slow.

**Predicted rows:** for each of the {U} unsold cars we calculate the Fast score and the Slow score, pick the bigger one, and count.
Then we divide each count by {U}:

- Fast: {pc['Fast']} ÷ {U} = {pc['Fast'] / U:.4f} = **{pc['Fast'] / U * 100:.1f}%**
- Slow: {pc['Slow']} ÷ {U} = {pc['Slow'] / U:.4f} = **{pc['Slow'] / U * 100:.1f}%**
- Cannot classify: {pc['Cannot classify']} ÷ {U} = {pc['Cannot classify'] / U:.4f} = **{pc['Cannot classify'] / U * 100:.1f}%**
"""
    )
    show(pl.results_split(counts, pc))

    st.subheader("Scores for the unsold cars, grouped by features (Table A2)")
    g = nb.group_predictions(pred)
    g_show = g.rename(columns={"Mileage_Band": "Mileage"}).copy()
    g_show["Fast score"] = g_show["Fast score"].map(lambda x: f"{x:.5f}")
    g_show["Slow score"] = g_show["Slow score"].map(lambda x: f"{x:.5f}")
    st.dataframe(g_show[["Type", "Mileage", "Condition", "Cars", "Fast score", "Slow score", "Prediction"]], hide_index=True)
    st.caption("Counting the groups: " + " + ".join(str(x) for x in g[g["Prediction"] == "Fast"]["Cars"]) + f" = {pc['Fast']} Fast;  "
               + " + ".join(str(x) for x in g[g["Prediction"] == "Slow"]["Cars"]) + f" = {pc['Slow']} Slow;  "
               + " + ".join(str(x) for x in g[g["Prediction"] == "Cannot classify"]["Cars"]) + f" = {pc['Cannot classify']} Cannot classify.")

    with st.expander("Show the full calculation for every group"):
        st.write(f"The {len(pred)} unsold cars have only {len(g)} different feature combinations, so we calculate each combination once. "
                 "For each class we multiply four fractions: the prior and the three likelihoods "
                 "(type, mileage band, condition). The bigger score is the prediction.")
        calc_rows = []
        for _, r in g.iterrows():
            res = nb.nb_scores(sold, {f: r[f] for f in nb.FEATURES})

            def calc(cls):
                fac = res[cls]["factors"]
                return " × ".join(f"{x['num']}/{x['den']}" for x in fac) + f" = {res[cls]['score']:.5f}"

            calc_rows.append({"Cars": r["Cars"], "Type": r["Type"], "Mileage": r["Mileage_Band"], "Condition": r["Condition"],
                              "Fast calculation": calc("Fast"), "Slow calculation": calc("Slow"), "Prediction": r["Prediction"]})
        st.dataframe(pd.DataFrame(calc_rows), hide_index=True)
        st.caption("Fraction order: prior × P(type | class) × P(mileage | class) × P(condition | class).")

    with st.expander("What does “Cannot classify” mean?"):
        st.write("The model picks the class with the bigger score. For these cars, both scores are 0, so there is nothing to compare. "
                 "A score is a product, and one 0 makes the whole product 0. These cars have a type that never appears in the sold cars "
                 "(no truck or hatchback was sold), so P(type | Fast) = 0 ÷ 28 = 0 and P(type | Slow) = 0 ÷ 35 = 0. "
                 "This does not mean the car will sell fast or slow. The data has no information about that car type.")

    with st.expander("See every unsold car"):
        allc = pred.copy()
        allc["Fast score"] = allc["Fast score"].map(lambda x: f"{x:.5f}")
        allc["Slow score"] = allc["Slow score"].map(lambda x: f"{x:.5f}")
        st.dataframe(allc, hide_index=True)

    st.subheader("Check any unsold car step by step")
    pick = st.selectbox("Choose a car", list(pred["Vehicle_ID"]), key="res_car")
    row = pred[pred["Vehicle_ID"] == pick].iloc[0]
    render_nb(nb.nb_scores(sold, {f: row[f] for f in nb.FEATURES}))

# ============================================================================
# TAB 7: Which features matter
# ============================================================================
with tabs[6]:
    st.header("7. Which features matter?")
    st.write("For each feature value we count how many sold cars sold within 30 days. "
             f"The overall fast-sale rate is {counts['Fast']} ÷ {N} = {counts['Fast'] / N * 100:.1f}%. "
             "A value far from this line is more useful for predicting.")
    rates = nb.feature_rates(sold)
    show(pl.rates_chart(rates, counts["Fast"] / N * 100))
    r = rates.copy()
    r["Calculation"] = r.apply(lambda x: f"{x['Fast']} ÷ {x['Sold cars']}", axis=1)
    r["Fast-sale rate (%)"] = r["Fast-sale rate (%)"].map(lambda x: f"{x:.1f}%")
    st.dataframe(r[["Feature", "Value", "Fast", "Sold cars", "Calculation", "Fast-sale rate (%)"]], hide_index=True)
    st.info("Vehicle type is the most useful feature (sedans 64%, SUVs 31%). Groups with very few cars "
            "(Very High mileage, Coupe, Fair condition) are too small to trust.")