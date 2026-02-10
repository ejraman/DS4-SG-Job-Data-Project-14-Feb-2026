import os, json, ast, re, textwrap
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns

from bokeh.plotting import figure
import holoviews as hv
from holoviews import opts
hv.extension("bokeh")

import statsmodels.formula.api as smf

# Optional: for Holoviews chord inside Streamlit (if installed)
try:
    from streamlit_bokeh import streamlit_bokeh
    _HAS_STREAMLIT_BOKEH = True
except Exception:
    _HAS_STREAMLIT_BOKEH = False

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="SG Jobs & Salary Dashboard (Max Features)", layout="wide")

DATA_PATH = "SGJobData.csv"

# -------------------------
# Helpers
# -------------------------
def parse_categories_any(cat_str):
    """
    Robust category parser:
    - Handles JSON strings (json.loads)
    - Handles python-literal strings (ast.literal_eval) as seen in some exports
    Returns list of dicts (possibly empty).
    """
    if pd.isna(cat_str):
        return []
    s = str(cat_str).strip()
    if not s:
        return []
    if s in ("[]", "nan", "None"):
        return []
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []

def first_category_name(cat_str):
    items = parse_categories_any(cat_str)
    if items and isinstance(items[0], dict) and "category" in items[0]:
        return items[0].get("category") or "Unknown"
    return "Unknown"

def clean_and_wrap_company(name, width=16):
    clean_name = re.sub(r'(?i)\s+PTE\.?\s+LTD\.?', '', str(name)).strip()
    return "<br>".join(textwrap.wrap(clean_name, width=width)) if clean_name else "Unknown"

@st.cache_data(show_spinner=False)
def load_and_prepare(path: str, nrows=None):
    # Read with pandas; load many columns (feature-rich app needs them)
    df = pd.read_csv(path, nrows=nrows, quotechar='"', doublequote=True)

    # Normalize key categorical columns
    for col in ["employmentTypes", "positionLevels", "postedCompany_name", "status_jobStatus", "title", "salary_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
        else:
            df[col] = "Unknown"

    # Numeric columns (support both naming conventions)
    for c in ["salary_minimum", "salary_maximum", "average_salary", "metadata_totalNumberOfView",
              "metadata_totalNumberJobApplication", "minimumYearsExperience"]:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace(r'[$,\s]', '', regex=True)
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    if "average_salary" in df.columns and df["average_salary"].notna().any():
        df["avg_salary"] = df["average_salary"]
    else:
        if df["salary_minimum"].notna().any() and df["salary_maximum"].notna().any():
            df["avg_salary"] = (df["salary_minimum"] + df["salary_maximum"]) / 2
        else:
            df["avg_salary"] = df["salary_maximum"]

    if "metadata_originalPostingDate" in df.columns:
        df["metadata_originalPostingDate"] = pd.to_datetime(df["metadata_originalPostingDate"], errors="coerce")

    if "categories" in df.columns:
        df["category"] = df["categories"].map(first_category_name)
        df["categories_list"] = df["categories"].map(parse_categories_any)
        df_exploded = df.explode("categories_list").reset_index(drop=True)
        df_exploded["category_name"] = df_exploded["categories_list"].apply(
            lambda x: x.get("category") if isinstance(x, dict) else "Unknown"
        )
    else:
        df["category"] = "Unknown"
        df_exploded = pd.DataFrame()

    if "salary_type" in df.columns:
        mask_monthly = df["salary_type"].fillna("").str.contains("Monthly", case=False, na=False)
        if mask_monthly.any():
            df = df[mask_monthly].copy()
            if not df_exploded.empty:
                # keep exploded aligned (best-effort)
                df_exploded = df_exploded.loc[df.index.intersection(df_exploded.index)].copy()

    df = df.dropna(subset=["avg_salary"])
    df = df[df["avg_salary"] > 0]

    return df, df_exploded

def kpi_block(dff):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Jobs", f"{len(dff):,}")
    c2.metric("Median Salary", f"${dff['avg_salary'].median():,.0f}")
    c3.metric("Mean Salary", f"${dff['avg_salary'].mean():,.0f}")
    c4.metric("Avg Views", f"{pd.to_numeric(dff.get('metadata_totalNumberOfView', pd.Series(dtype=float)), errors='coerce').mean():.1f}")
    c5.metric("Avg Applications", f"{pd.to_numeric(dff.get('metadata_totalNumberJobApplication', pd.Series(dtype=float)), errors='coerce').mean():.2f}")

# -------------------------
# Load data
# -------------------------
st.title("🇸🇬 Job & Salary Transparency Dashboard (Max Features)")

if not os.path.exists(DATA_PATH):
    st.error(f"⚠️ File '{DATA_PATH}' cannot be found in the current directory.")
    st.info("Put SGJobData.csv in the same folder as this app, then rerun.")
    st.stop()

with st.spinner("Loading data (cached after first load)..."):
    df, df_exploded = load_and_prepare(DATA_PATH, nrows=None)  # set nrows=200000 if you need speed

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")

cat_options = ["All"] + sorted(df["category"].dropna().unique().tolist())
emp_options = ["All"] + sorted(df["employmentTypes"].dropna().unique().tolist())
lvl_options = ["All"] + sorted(df["positionLevels"].dropna().unique().tolist())

sel_cat = st.sidebar.selectbox("Category", cat_options, index=0)
sel_emp = st.sidebar.selectbox("Employment Type", emp_options, index=0)
sel_lvl = st.sidebar.selectbox("Position Level", lvl_options, index=0)

exp_max = float(np.nanmax(pd.to_numeric(df["minimumYearsExperience"], errors="coerce").fillna(0).values))
sel_exp = st.sidebar.slider(
    "Min Years Experience",
    min_value=0.0,
    max_value=max(0.0, exp_max),
    value=(0.0, min(10.0, exp_max)),
    step=0.5
)

sal_min = float(np.nanpercentile(df["avg_salary"].values, 1))
sal_max = float(np.nanpercentile(df["avg_salary"].values, 99))
sel_sal = st.sidebar.slider(
    "Monthly Salary (Avg)",
    min_value=float(sal_min),
    max_value=float(sal_max),
    value=(float(sal_min), float(min(sal_max, sal_min + 8000))),
    step=100.0
)

company_kw = st.sidebar.text_input("Company contains", value="").strip().lower()

# Apply filters
dff = df.copy()
if sel_cat != "All":
    dff = dff[dff["category"] == sel_cat]
if sel_emp != "All":
    dff = dff[dff["employmentTypes"] == sel_emp]
if sel_lvl != "All":
    dff = dff[dff["positionLevels"] == sel_lvl]

dff = dff[(dff["minimumYearsExperience"].fillna(0) >= sel_exp[0]) &
          (dff["minimumYearsExperience"].fillna(0) <= sel_exp[1])]

dff = dff[(dff["avg_salary"] >= sel_sal[0]) & (dff["avg_salary"] <= sel_sal[1])]

if company_kw:
    dff = dff[dff["postedCompany_name"].fillna("").str.lower().str.contains(company_kw)]

# -------------------------
# KPI
# -------------------------
kpi_block(dff)
st.divider()

# -------------------------
# Tabs
# -------------------------
tab_overview, tab_benchmark, tab_categories, tab_advanced, tab_model = st.tabs(
    ["Overview", "Benchmarking", "Categories", "Advanced Visuals", "Statsmodels"]
)

with tab_overview:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Salary Distribution")
        fig = px.histogram(dff, x="avg_salary", nbins=60, title="Average Monthly Salary Distribution")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

        fig_box = px.box(dff, y="avg_salary", points="outliers", title="Salary Spread (Box Plot)")
        fig_box.update_layout(height=300)
        st.plotly_chart(fig_box, use_container_width=True)

    with right:
        st.subheader("Market Demand")
        top_cat = dff["category"].value_counts().head(15).reset_index()
        top_cat.columns = ["category", "job_count"]
        fig_cat = px.bar(top_cat, x="job_count", y="category", orientation="h", title="Top Categories (Job Count)")
        fig_cat.update_layout(height=330)
        st.plotly_chart(fig_cat, use_container_width=True)

        top_co = dff["postedCompany_name"].value_counts().head(15).reset_index()
        top_co.columns = ["company", "job_count"]
        fig_co = px.bar(top_co, x="job_count", y="company", orientation="h", title="Top Companies (Job Count)")
        fig_co.update_layout(height=330)
        st.plotly_chart(fig_co, use_container_width=True)

    st.subheader("Salary vs Experience")
    sample_n = min(30000, len(dff))
    dffs = dff.sample(sample_n, random_state=42) if sample_n > 0 else dff
    fig_scatter = px.scatter(
        dffs,
        x="minimumYearsExperience",
        y="avg_salary",
        hover_data=["title", "postedCompany_name", "category", "positionLevels"],
        trendline="ols",
        title="Salary vs Minimum Years Experience (with OLS trendline)"
    )
    fig_scatter.update_layout(height=520)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Raw Table (Top 200)")
    cols = [c for c in [
        "title","postedCompany_name","category","employmentTypes","positionLevels",
        "minimumYearsExperience","salary_minimum","salary_maximum","avg_salary",
        "metadata_totalNumberOfView","metadata_totalNumberJobApplication"
    ] if c in dff.columns]
    st.dataframe(dff[cols].head(200), use_container_width=True)

with tab_benchmark:
    st.subheader("Salary Benchmarking (Median by Company within Position Level)")

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        user_pos = st.selectbox("Position Level:", options=sorted(df["positionLevels"].unique()), index=0)
    with col_in2:
        user_sal = st.number_input("Your Monthly Salary ($):", value=6000, step=500)

    filtered_df = df[df["positionLevels"] == user_pos].copy()

    if {"postedCompany_name", "positionLevels", "avg_salary"}.issubset(filtered_df.columns):
        filtered_df = filtered_df.drop_duplicates(subset=["postedCompany_name", "positionLevels", "avg_salary"])

    if any(k in user_pos for k in ["Fresh", "Junior", "Entry"]):
        filtered_df = filtered_df[filtered_df["avg_salary"] < 10000]

    filtered_df = filtered_df[filtered_df["avg_salary"] > 0]

    pos_data = (filtered_df.groupby("postedCompany_name")["avg_salary"]
                .median()
                .reset_index()
                .sort_values("avg_salary")
                .tail(20))

    fig_bench = px.line(
        pos_data,
        x="postedCompany_name",
        y="avg_salary",
        title=f"Market Median for {user_pos} (Top 20 Companies by Median)",
        markers=True,
        labels={"postedCompany_name": "Company", "avg_salary": "Median Monthly Salary ($)"}
    )
    fig_bench.add_hline(
        y=user_sal,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Your Salary: ${user_sal:,.0f}"
    )
    fig_bench.update_layout(height=420, xaxis_showticklabels=False, hovermode="x unified",
                            margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_bench, use_container_width=True)

    st.subheader("Top 10 Paying Companies (Mean salary within selected level)")
    top_10_pay = (df[df["positionLevels"] == user_pos]
                  .groupby("postedCompany_name")["avg_salary"]
                  .mean()
                  .nlargest(10)
                  .reset_index())
    top_10_pay["Company"] = top_10_pay["postedCompany_name"].apply(clean_and_wrap_company)

    fig_top_pay = px.bar(
        top_10_pay,
        x="avg_salary",
        y="Company",
        orientation="h",
        color="avg_salary",
        color_continuous_scale="Viridis",
        labels={"avg_salary": "Average Salary ($)", "Company": ""}
    )
    fig_top_pay.update_layout(height=520, margin=dict(l=140, r=20, t=30, b=50),
                              yaxis={"categoryorder": "total ascending", "title": None, "automargin": True},
                              xaxis_title="Average Salary ($)",
                              coloraxis_colorbar=dict(title="Average Salary"))
    st.plotly_chart(fig_top_pay, use_container_width=True)

    st.subheader("Salary Distribution by Position Level (Violin, capped at $30k)")
    v_data = df[(df["avg_salary"] > 0) & (df["avg_salary"] < 30000)].copy()
    sns.set_theme(style="whitegrid")
    fig_v, ax_v = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=v_data,
        x="positionLevels",
        y="avg_salary",
        hue="positionLevels",
        palette="bright",
        ax=ax_v,
        legend=False
    )
    ax_v.set_title("Salary Density by Position Level (Up to $30k)")
    ax_v.set_ylabel("Monthly Salary ($)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig_v, use_container_width=True)

with tab_categories:
    if df_exploded.empty:
        st.warning("No categories data available (missing or unparsable 'categories' column).")
    else:
        st.subheader("Top 10 Job Categories: Viewership & Salary (Sunburst)")

        top_cats = df_exploded["category_name"].value_counts().nlargest(10).index
        df_filtered = df_exploded[df_exploded["category_name"].isin(top_cats)].copy()

        df_filtered["salary_maximum"] = pd.to_numeric(df_filtered.get("salary_maximum", np.nan), errors="coerce").fillna(0)
        df_filtered["metadata_totalNumberOfView"] = pd.to_numeric(df_filtered.get("metadata_totalNumberOfView", np.nan), errors="coerce").fillna(0)

        fig1 = px.sunburst(
            df_filtered,
            path=["category_name", "status_jobStatus"],
            values="metadata_totalNumberOfView",
            color="salary_maximum",
            color_continuous_scale="RdBu",
            range_color=[3000, 9000],
            color_continuous_midpoint=6000,
            labels={"salary_maximum": "Max Salary"}
        )
        fig1.update_coloraxes(
            colorbar_tickvals=[0, 3000, 6000, 9000, 12000],
            colorbar_ticktext=["0k", "3k", "6k", "9k", "12k"],
            colorbar_title="Max Salary"
        )
        fig1.update_layout(height=650)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Interactive Sunburst (Top 10 Companies, custom hierarchy + salary ring)")
        full_list = ["employmentTypes", "positionLevels", "postedCompany_name", "status_jobStatus", "category"]
        selectable = [c for c in full_list if c in df.columns and c not in ("status_jobStatus", "employmentTypes")]
        default_fields = [c for c in ["postedCompany_name", "positionLevels"] if c in selectable]
        path_selection = st.multiselect(
            "Select hierarchy fields (inner ➜ outer):",
            options=selectable,
            default=default_fields
        )

        if path_selection:
            top_10_companies = df["postedCompany_name"].value_counts().nlargest(10).index
            df_c2 = df[df["postedCompany_name"].isin(top_10_companies)].copy()
            df_c2["salary_label"] = (df_c2["avg_salary"] / 1000).round(0).astype(int).astype(str) + "k"
            final_path = path_selection + ["salary_label"]

            fig2 = px.sunburst(
                df_c2,
                path=final_path,
                values="avg_salary",
                color="avg_salary",
                color_continuous_scale="RdBu",
                range_color=[0, 12000],
                maxdepth=4
            )
            fig2.update_coloraxes(
                colorbar_tickvals=[0, 3000, 6000, 9000, 12000],
                colorbar_ticktext=["0k", "3k", "6k", "9k", "12k"],
                colorbar_title="Avg Salary"
            )
            fig2.update_layout(height=700, margin=dict(t=40, l=20, r=20, b=20))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Select at least one field to build the sunburst hierarchy.")

with tab_advanced:
    st.subheader("Interactive Histogram (color by selected field)")
    safe_options = [c for c in ["employmentTypes", "positionLevels", "status_jobStatus", "category"] if c in df.columns]
    sel_col = st.selectbox("Color by:", options=safe_options, index=0)
    fig3 = px.histogram(dff, x="avg_salary", color=sel_col, range_x=[0, 20000])
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Market Momentum: Pay vs Visibility (LOWESS trend)")
    seaborn_colors = sns.color_palette("bright", 10).as_hex()
    df_sampled = dff.iloc[::5].copy() if len(dff) > 0 else dff
    if len(df_sampled) > 0 and "metadata_totalNumberOfView" in df_sampled.columns:
        fig4 = px.scatter(
            df_sampled,
            x="avg_salary",
            y="metadata_totalNumberOfView",
            color="positionLevels",
            trendline="lowess",
            trendline_scope="overall",
            trendline_color_override="black",
            color_discrete_sequence=seaborn_colors,
            render_mode="svg",
            title="Market Momentum (Pay vs Views)"
        )
        fig4.update_traces(marker=dict(size=7, opacity=0.5))
        fig4.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True, config={"scrollZoom": True})
    else:
        st.info("No data after filters, or views column missing.")

    st.subheader("Bokeh Histogram (Alt view)")
    vals = dff["avg_salary"].dropna().values
    if len(vals) > 0:
        hist, edges = np.histogram(vals, bins=50)
        p = figure(height=320, title="Bokeh Histogram: Avg Salary", tools="pan,wheel_zoom,box_zoom,reset,save")
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
        st.bokeh_chart(p, use_container_width=True)

    st.subheader("Holoviews: Top 20 Categories (Median Salary)")
    agg = (dff.groupby("category", as_index=False)
           .agg(median_salary=("avg_salary", "median"), job_count=("avg_salary", "size"))
           .sort_values("job_count", ascending=False)
           .head(20))
    if len(agg) > 0:
        hv_plot = hv.Bars(agg, kdims="category", vdims="median_salary").opts(
            height=360, width=1000, xrotation=45, tools=["hover"], title="Top 20 Categories: Median Salary"
        )
        st.write(hv_plot)

    st.subheader("Salary Flow (Chord): Position Levels vs Job Titles (Top 15 titles)")
    if not _HAS_STREAMLIT_BOKEH:
        st.info("Optional feature: install `streamlit-bokeh` to display the chord interactively.")
    else:
        top_titles_list = df["title"].value_counts().nlargest(15).index
        df_chord_data = df[df["title"].isin(top_titles_list)]
        chord_df = (df_chord_data.groupby(["positionLevels", "title"])["avg_salary"]
                    .mean()
                    .reset_index())
        chord_df.columns = ["source", "target", "value"]

        nodes_list = hv.Dataset(pd.concat([chord_df["source"], chord_df["target"]]).unique(), "index")
        chord_chart = hv.Chord((chord_df, nodes_list)).opts(
            opts.Chord(
                cmap="Category20",
                edge_color=hv.dim("source").str(),
                labels="index",
                node_color=hv.dim("index").str(),
                width=750,
                height=750,
                label_text_font_size="10pt",
                edge_alpha=0.7,
                node_size=12,
                tools=["hover"],
                title="Salary Flow: Level ➜ Title"
            )
        )
        chord_rendered = hv.render(chord_chart, backend="bokeh")
        streamlit_bokeh(chord_rendered, use_container_width=True, key="salary_chord")

with tab_model:
    st.subheader("Statsmodels: What drives salary?")
    st.caption("Quick OLS regression (sampled). Coefficients are directional, not causal.")

    model_sample_n = min(50000, len(dff))
    if model_sample_n < 1000:
        st.warning("Too little data after filters to run regression. Try widening filters.")
    else:
        mdf = dff.sample(model_sample_n, random_state=7).copy()
        mdf = mdf.dropna(subset=["avg_salary", "minimumYearsExperience", "positionLevels", "category"])
        mdf["log_salary"] = np.log(mdf["avg_salary"])

        top_cats = mdf["category"].value_counts().head(10).index
        mdf = mdf[mdf["category"].isin(top_cats)]

        formula = "log_salary ~ minimumYearsExperience + C(positionLevels) + C(category)"
        try:
            res = smf.ols(formula, data=mdf).fit()
            st.text(res.summary().as_text()[:4000])

            st.subheader("Top Coefficients (by absolute effect)")
            coef = res.params.drop("Intercept", errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
            st.dataframe(coef.head(20).rename("coef").to_frame(), use_container_width=True)
        except Exception as e:
            st.error(f"Regression failed: {e}")
