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
st.set_page_config(page_title="Singapore Jobs Market Salary Dashboard for Jobseekers", layout="wide")

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
st.title("Singapore Jobs Market Salary Dashboard for Jobseekers")

if not os.path.exists(DATA_PATH):
    st.error(f"⚠️ File '{DATA_PATH}' cannot be found in the current directory.")
    st.info("Put SGJobData.csv in the same folder as this app, then rerun.")
    st.stop()

with st.spinner("Loading data (cached after first load)..."):
    df, df_exploded = load_and_prepare(DATA_PATH, nrows=None)  # set nrows=200000 if you need speed

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 400px;
        max-width: 400px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
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
    max_value=20.0,  # Capped at 20 as requested
    value=2.0,       # Default starting point
    step=0.5,        # Enables 0.5, 1.0, 1.5 increments
    format="%.1f"    # Shows 1 decimal place (e.g., 1.5)
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

# --- CHANGE THIS SECTION ---
# dff = dff[(dff["minimumYearsExperience"].fillna(0) >= sel_exp[0]) &
#           (dff["minimumYearsExperience"].fillna(0) <= sel_exp[1])]

# --- TO THIS ---
# Filter for jobs requiring UP TO the experience selected in the sidebar
dff = dff[dff["minimumYearsExperience"].fillna(0) <= sel_exp]
dff = dff[(dff["avg_salary"] >= sel_sal[0]) & (dff["avg_salary"] <= sel_sal[1])]

if company_kw:
    dff = dff[dff["postedCompany_name"].fillna("").str.lower().str.contains(company_kw)]

# -------------------------
# KPI
# -------------------------
kpi_block(dff)
st.divider()

# 2. INSERT YOUR MODEL ENGINE BLOCK HERE
model_df = dff.dropna(subset=["avg_salary", "minimumYearsExperience", "positionLevels", "category"]).copy()
model_df.columns = [c.replace(" ", "_") for c in model_df.columns] 

intercept, exp_coef, level_params, cat_params = 0, 0, {}, {}
model_ready = False

if len(model_df) > 50:
    try:
        # Use C() to handle the categorical text data
        formula = "avg_salary ~ minimumYearsExperience + C(positionLevels) + C(category)"
        model_obj = smf.ols(formula=formula, data=model_df).fit()
        intercept = model_obj.params['Intercept']
        exp_coef = model_obj.params['minimumYearsExperience']
        level_params = {k: v for k, v in model_obj.params.items() if "positionLevels" in k}
        cat_params = {k: v for k, v in model_obj.params.items() if "category" in k}
        model_ready = True
    except: 
        model_ready = False
# -------------------------
# Tabs
# -------------------------
# -------------------------
# Tabs Section (FIXED INDENTATION)
# -------------------------
tab_salarybenchmarking, tab_careerplanning, tab_model = st.tabs(
    ["Salary Benchmarking", "Career Planning", "Statsmodels"]
)

with tab_salarybenchmarking:
    st.header("🧮 Personal Fair-Pay Estimator")
    
    # Check if Sidebar filters are specific enough for a prediction
    # Logic: We need a specific Level and Category (not "All")
    if sel_cat != "All" and sel_lvl != "All":
        if model_ready:
            # We use the lower bound of your sidebar experience slider: sel_exp[0]
            #user_exp_val = sel_exp[0] 
            user_exp_val = sel_exp     # Simply use the variable directly

            # PREDICTION LOGIC: Base + (Years * Rate) + Level Premium + Category Premium
            pred = intercept + (user_exp_val * exp_coef)
            
            # Add Level Premium from model coefficients
            lvl_key = f"C(positionLevels)[T.{sel_lvl}]"
            if lvl_key in level_params:
                pred += level_params[lvl_key]

            # Add Category Premium from model coefficients
            cat_key = f"C(category)[T.{sel_cat}]"
            if cat_key in cat_params:
                pred += cat_params[cat_key]

            # Display the high-impact result
            st.metric(
                label=f"Predicted Market Rate for {sel_lvl} in {sel_cat}", 
                value=f"S$ {max(0, pred):,.0f} / month",
                delta=f"Based on {user_exp_val} years exp",
                delta_color="off"
            )
            st.caption("💡 *This estimate uses OLS regression to isolate the impact of your seniority and industry.*")
        else:
            st.warning("⚠️ The model is still warming up. Try widening your sidebar filters to include more data.")
    else:
        st.info("👈 **Action Required:** Please select a specific **Category** and **Position Level** in the sidebar to calculate your personalized Fair-Pay estimate.")

    st.divider()

    # Role-Specific Pay Bands (Violin) - Much more useful for Jobseekers
    st.subheader(f"Detailed Pay Bands for {sel_cat if sel_cat != 'All' else 'the Market'}")
    
    # Filter for the violin (keep under 30k for better scale)
    v_data = dff[dff['avg_salary'] < 30000].copy()
    
    if not v_data.empty:
        fig_violin = px.violin(
            v_data, 
            y='avg_salary', 
            x='positionLevels', 
            color='positionLevels',         
            box=True,             
            points=False, 
            title="Distribution of Actual Job Listings",
            category_orders={"positionLevels": sorted(df['positionLevels'].unique())}
        )
        fig_violin.update_layout(
            height=500, 
            showlegend=False, 
            xaxis_title="Seniority Level",
            yaxis_title="Monthly Salary (SGD)",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.warning("No listings found matching your current sidebar filters.")

    st.subheader("Company Comp-Intelligence")
    full_list = ["employmentTypes", "positionLevels", "postedCompany_name", "category"]
    selectable = [c for c in full_list if c in df.columns]
    path_selection = st.multiselect("Select hierarchy fields:", options=selectable, default=["postedCompany_name", "positionLevels"])

    if path_selection:
        top_10 = dff["postedCompany_name"].value_counts().nlargest(10).index
        df_c2 = dff[dff["postedCompany_name"].isin(top_10)].copy()
        df_c2["salary_label"] = (df_c2["avg_salary"] / 1000).round(0).astype(int).astype(str) + "k"
        
        fig2 = px.sunburst(
            df_c2,
            path=path_selection + ["salary_label"],
            values="avg_salary",
            color="avg_salary",
            color_continuous_scale="RdBu",
            range_color=[0, 12000],
            maxdepth=4
        )
        fig2.update_layout(height=850, margin=dict(t=20, l=0, r=0, b=0),uniformtext=dict(minsize=10, mode='hide'))
        st.plotly_chart(fig2, use_container_width=True)

# Important: The next 'with' statement must be back at the main indentation level
with tab_careerplanning:
    st.subheader("Experience-to-Earnings Trajectory (OLS Trendline)")
    
    # 1. Logic: Use the filtered data (dff) from your sidebar
    if len(dff) > 0:
        sample_n = min(10000, len(dff)) 
        dff_scatter = dff.sample(sample_n, random_state=42)

        # 2. Create the Scatter Plot
        fig_scatter = px.scatter(
            dff_scatter,
            x="minimumYearsExperience",
            y="avg_salary",
            color="category", 
            hover_data=["title", "postedCompany_name", "category", "positionLevels"],
            trendline="ols",
            title=f"Salary Trend: {sel_cat} | {sel_lvl}",
            labels={
                "minimumYearsExperience": "Years of Experience",
                "avg_salary": "Monthly Salary (SGD)",
                "category": "Industry"
            }
        )

        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("No data matches your sidebar filters.")

    st.divider()

    st.subheader(f"Career Path & Mobility Flow: Position Levels vs Job Titles (Top 15 titles)")
    if not _HAS_STREAMLIT_BOKEH:
        st.info("Optional feature: install `streamlit-bokeh` to display the chord interactively.")
    else:
        if len(dff) > 0:
            # 1. Prepare Chord Data
            top_titles_list = dff["title"].value_counts().nlargest(15).index
            df_chord_data = dff[dff["title"].isin(top_titles_list)]
            
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
                    title=f"Salary Flow for {sel_cat}"
                )
            )

            # 2. THE LAYOUT (3:1 ratio for chart vs insights)
            chord_col, summary_col = st.columns([3, 1]) 

            with chord_col:
                # Render the Chord Chart on the left
                chord_rendered = hv.render(chord_chart, backend="bokeh")
                
                # FIX: Use a unique key based on the current selection to prevent Duplicate Key errors
                dynamic_key = f"chord_{sel_lvl}_{sel_cat}".replace(" ", "_")
                streamlit_bokeh(chord_rendered, use_container_width=True, key=dynamic_key)

            with summary_col:
                # Show Trending Insights on the right
                st.markdown("### 🔥 Market Hits")
                st.caption(f"Top roles for **{sel_lvl}** in **{sel_cat}**:")
                
                # Extract Top 3 most frequent titles from filtered data
                top_3 = dff['title'].value_counts().head(3)
                
                if not top_3.empty:
                    for title, count in top_3.items():
                        # Calculate avg salary for this specific title
                        avg_pay = dff[dff['title'] == title]['avg_salary'].mean()
                        
                        st.metric(
                            label=textwrap.shorten(title, width=30, placeholder="..."), 
                            value=f"{count} Jobs", 
                            delta=f"${avg_pay:,.0f} Avg", 
                            delta_color="normal"
                        )
                        st.write("---")
                else:
                    st.info("Narrow filters to see top roles.")
            
            # --- THE DUPLICATE BLOCK PREVIOUSLY HERE HAS BEEN REMOVED ---
            
        else:
            st.info("Adjust filters to see career flow.")


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
        # ZWX 1. OMIT
            #st.text(res.summary().as_text()[:4000])

            #st.subheader("Top Coefficients (by absolute effect)")
            #coef = res.params.drop("Intercept", errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
            #st.dataframe(coef.head(20).rename("coef").to_frame(), use_container_width=True)

            # ZWX 1. OMIT THE NOISY TEXT: Hide the summary() and only show R-squared (The "Accuracy")
            st.metric("Model Prediction Strength (R²)", f"{res.rsquared:.1%}", 
                      help="This shows how much of the salary variation is explained by Level, Category, and Experience.")

            # 2. TRANSFORM DATA: Get top coefficients and convert log-change to percentage
            # Math: (exp(coef) - 1) * 100 gives the percentage salary boost
            coef_df = res.params.drop("Intercept", errors="ignore").to_frame(name="raw_coef")
            coef_df["Salary Boost (%)"] = (np.exp(coef_df["raw_coef"]) - 1) * 100
            coef_df = coef_df.sort_values("Salary Boost (%)", ascending=False).head(15)

            # 3. VISUALIZE: Create a high-impact bar chart
            st.subheader("🚀 The Top 'Salary Boosters'")
            import plotly.express as px
            fig_boost = px.bar(
                coef_df, 
                x="Salary Boost (%)", 
                y=coef_df.index, 
                orientation='h',
                color="Salary Boost (%)",
                color_continuous_scale="Viridis",
                title="Impact on Salary (Relative to Entry Level/General Category)"
            )
            fig_boost.update_layout(height=500, yaxis_title=None, showlegend=False)
            st.plotly_chart(fig_boost, use_container_width=True)

            # 4. DATA AUDIT: Keep the clean dataframe hidden in an expander for Q&A
            with st.expander("View Technical Coefficients (Data Audit)"):
                st.dataframe(coef_df[["Salary Boost (%)"]].style.format("{:.1f}%"), use_container_width=True)
        except Exception as e:
            st.error(f"Regression failed: {e}")
