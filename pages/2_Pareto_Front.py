"""
App 2 — Pareto Front for Multi-Objective Optimization
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

st.set_page_config(page_title="Pareto Front", layout="wide")

st.title("Multi-Objective Optimization — Pareto Front")
st.markdown("Upload your own dataset **or** use the built-in smartphone example to explore the Pareto front.")

# ── Pareto logic ──────────────────────────────────────────────────────────────
def find_pareto(costs):
    """
    costs: 2D array where each column is an objective to MINIMIZE.
    Returns boolean mask — True = Pareto optimal.
    """
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        dominated[i] = False
        is_efficient[dominated] = False
    return is_efficient

# ── Built-in dataset ──────────────────────────────────────────────────────────
DEFAULT_DATA = {
    "Model_Name":        ["Alpha X1","Beta Pro","Gamma S","Delta Max","Epsilon Lite",
                          "Zeta Ultra","Eta Plus","Theta Go","Iota Edge","Kappa Air",
                          "Lambda Z","Mu Prime","Nu Fast","Xi Budget","Omicron SE"],
    "Price_USD":         [999,  799,  649,  1199, 349,  1099, 549, 299, 899, 749,
                          1299, 449,  599,  199,  849],
    "Performance_Score": [92,   85,   78,   97,   55,   95,   72,  48,  88,  82,
                          99,   65,   76,   42,   80],
    "Battery_Hours":     [12,   14,   10,   11,   18,   9,    13,  20,  10,  15,
                          8,    16,   11,   22,   13],
    "Camera_MP":         [108,  64,   48,   200,  12,   108,  50,  8,   64,  48,
                          200,  16,   50,   5,    64],
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    use_default = st.radio("Dataset", ["Use built-in smartphone data", "Upload CSV"])
    uploaded = None
    if use_default == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    st.header("Objectives")
    st.caption("Select two columns and their optimization direction.")

# ── Load data ─────────────────────────────────────────────────────────────────
if use_default == "Upload CSV" and uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df)} rows from uploaded file.")
else:
    df = pd.DataFrame(DEFAULT_DATA)
    if use_default == "Upload CSV":
        st.info("No file uploaded — showing built-in smartphone data.")

st.subheader("Dataset Preview")
st.dataframe(df, use_container_width=True)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns.")
    st.stop()

with st.sidebar:
    obj1_col = st.selectbox("Objective 1 (X axis)", numeric_cols, index=0)
    obj1_dir = st.radio("Direction 1", ["Minimize", "Maximize"], index=0, horizontal=True)
    obj2_col = st.selectbox("Objective 2 (Y axis)", numeric_cols,
                             index=1 if len(numeric_cols) > 1 else 0)
    obj2_dir = st.radio("Direction 2", ["Minimize", "Maximize"], index=1, horizontal=True)
    label_col = st.selectbox("Label column (optional)",
                              ["(none)"] + df.select_dtypes(exclude=np.number).columns.tolist())
    run_btn = st.button("Find Pareto Front", type="primary", use_container_width=True)

st.markdown("---")

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    v1 = df[obj1_col].values.astype(float)
    v2 = df[obj2_col].values.astype(float)

    # Flip to minimize if needed
    c1 = v1 if obj1_dir == "Minimize" else -v1
    c2 = v2 if obj2_dir == "Minimize" else -v2
    costs = np.column_stack([c1, c2])

    mask = find_pareto(costs)
    df["Pareto"] = mask
    pareto_df   = df[mask].copy()
    dominated_df = df[~mask].copy()

    # ── Metrics ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Total models", len(df))
    col2.metric("Pareto optimal", int(mask.sum()))
    col3.metric("Dominated", int((~mask).sum()))

    # ── Plot ──────────────────────────────────────────────────────────────────
    st.subheader("Pareto Front Plot")
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(dominated_df[obj1_col], dominated_df[obj2_col],
               c="lightgrey", edgecolors="grey", s=80, alpha=0.7, label="Dominated", zorder=2)
    ax.scatter(pareto_df[obj1_col], pareto_df[obj2_col],
               c="crimson", edgecolors="black", s=120, label="Pareto Optimal", zorder=3)

    # Draw staircase front
    sort_col = obj1_col
    pf = pareto_df.sort_values(by=sort_col)
    ax.plot(pf[obj1_col], pf[obj2_col], "r--", alpha=0.6, linewidth=1.5, zorder=2)

    # Labels
    if label_col != "(none)":
        for _, row in pareto_df.iterrows():
            ax.annotate(str(row[label_col]),
                        (row[obj1_col], row[obj2_col]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color="darkred")

    ax.set_xlabel(f"{obj1_col}  [{obj1_dir}]", fontsize=12)
    ax.set_ylabel(f"{obj2_col}  [{obj2_dir}]", fontsize=12)
    ax.set_title("Pareto Front", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Pareto table ──────────────────────────────────────────────────────────
    st.subheader("Pareto Optimal Solutions")
    show_cols = ([label_col] if label_col != "(none)" else []) + [obj1_col, obj2_col]
    st.dataframe(pareto_df[show_cols].reset_index(drop=True), use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    csv_buf = io.StringIO()
    pareto_df.to_csv(csv_buf, index=False)
    st.download_button("Download Pareto solutions as CSV",
                       data=csv_buf.getvalue(),
                       file_name="pareto_solutions.csv",
                       mime="text/csv")
else:
    st.info("Configure objectives in the sidebar and click **Find Pareto Front**.")
    st.markdown("""
    **What is a Pareto Front?**

    A solution is *Pareto optimal* if you cannot improve one objective without making another worse.
    The Pareto front is the set of all such solutions — it represents the best possible trade-offs.

    **Example:** A smartphone with lower price *and* higher performance than another is better on both objectives.
    The dominated phone is removed from the front.
    """)
