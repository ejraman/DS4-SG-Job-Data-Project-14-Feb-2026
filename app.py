import os, ast, time, re 
import textwrap
import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import holoviews as hv
from holoviews import opts
# New import for the custom component
from streamlit_bokeh import streamlit_bokeh

# 1. SET PAGE TO WIDE MODE (Removes side whitespace)
st.set_page_config(layout="wide", page_title="Job Analysis Dashboard")

# ---------------- 2. CACHED DATA LOADING ---------------- #
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_path):
    # Step 1: Read File
    df = pd.read_csv(file_path, quotechar='"', doublequote=True)
    df = df.head(200000)

    # Step 2: Salary Cleaning
    numeric_cols = ['salary_maximum', 'average_salary', 'metadata_totalNumberOfView']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            #df.loc[df[col] > 50000, col] = 20000 

    # Step 3: Hierarchy Prep
    categorical_cols = ['employmentTypes', 'positionLevels', 'postedCompany_name', 'status_jobStatus']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # Step 4: Category Processing (This defines df_exploded)
    if 'categories' in df.columns:
        df['categories'] = df['categories'].fillna('[]').astype(str)
        # Convert string representation of list to actual list
        df['categories_list'] = df['categories'].apply(lambda x: ast.literal_eval(x) if x.startswith('[') else [])
        df_exploded = df.explode('categories_list').reset_index(drop=True)
        # Extract 'category' name from the dictionary
        df_exploded['category_name'] = df_exploded['categories_list'].apply(
            lambda x: x['category'] if isinstance(x, dict) else "Unknown"
        )
    else:
        df_exploded = pd.DataFrame()

    return df, df_exploded

# ---------------- 3. DATA SOURCE CHECK (NO SIDEBAR) ---------------- #
target_file = 'SGJobData.csv'

if not os.path.exists(target_file):
    # Direct error messaging as requested
    st.error(f"⚠️ File '{target_file}' cannot be found.")
    st.info("Please upload the file in the directory before rerun.")
    st.stop() 

# ---------------- 4. MAIN APP EXECUTION ---------------- #
# This now correctly receives TWO variables from the function
df_original, df_exploded = load_and_preprocess_data(target_file)

# Identify available columns
full_list = ['employmentTypes', 'positionLevels', 'postedCompany_name', 'status_jobStatus', 'average_salary']
actual_full = [opt for opt in full_list if opt in df_original.columns or opt == 'average_salary']
# Safe list (No Company) to prevent hanging in heavy charts
safe_options = [opt for opt in actual_full if opt != 'postedCompany_name']

# 1. Prepare the Seaborn high-contrast palette
# Create the color list from seaborn
seaborn_bright = sns.color_palette("bright", 12).as_hex()

# ---------------- 4. ROW 1: THREE COLUMNS ---------------- #
row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    st.subheader("1. Salary Benchmarking (Median of Avg Salary)")
    
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        # Get position from the CamelCase column in your Hierarchy Prep
        user_pos = st.selectbox("Position Level:", options=sorted(df_original['positionLevels'].unique()))
    with input_col2:
        user_sal = st.number_input("Monthly Salary ($):", value=2000, step=500)

    # --- THE MEDIAN CALCULATION LOGIC ---
    
    # 1. Filter by the selected Position Level first
    mask = df_original['positionLevels'] == user_pos
    filtered_df = df_original[mask].copy()

    # 2. DE-DUPLICATION: Remove identical job ads (Same Company + Level + Salary)
    # This ensures one company posting the same ad 10 times doesn't break the median
    filtered_df = filtered_df.drop_duplicates(subset=['postedCompany_name', 'positionLevels', 'average_salary'])

    # 3. SANITY FILTER: Remove obvious outliers for Fresh/Junior roles
    # (Prevents $10k+ "Fresh" salaries which are usually annual figures or data errors)
    if any(keyword in user_pos for keyword in ["Fresh", "Junior", "Entry"]):
        filtered_df = filtered_df[filtered_df['average_salary'] < 10000]
    
    # Ignore 0 values which pull the median down incorrectly
    filtered_df = filtered_df[filtered_df['average_salary'] > 0]

    # 4. GROUP BY COMPANY & CALCULATE MEDIAN
    # This takes all 'average_salary' values for the company and finds the middle one
    pos_data = (
        filtered_df.groupby('postedCompany_name')['average_salary']
        .median()
        .reset_index()
        .sort_values('average_salary')
    )

    # 5. Take the top 20 for the visualization
    pos_data = pos_data.tail(20)

    # --- PLOTLY VISUALIZATION ---
    fig_bench = px.line(
        pos_data, 
        x='postedCompany_name', 
        y='average_salary', 
        title=f"Market Median for {user_pos}",
        markers=True,
        labels={'postedCompany_name': 'Company', 'average_salary': 'Median of Avg Salary ($)'}
    )
    
    # Add your salary as a horizontal benchmark
    fig_bench.add_hline(
        y=user_sal, 
        line_dash="dot", 
        line_color="red", 
        annotation_text=f"Your Salary: ${user_sal:,.0f}"
    )
    
    # Layout adjustments: Hide messy X-labels (visible on hover)
    fig_bench.update_layout(
        height=400, 
        xaxis_showticklabels=False, 
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    st.plotly_chart(fig_bench, use_container_width=True)

with row1_col2:
    st.subheader("2. Top 10 Paying Companies")
    
    # --- STEP 1: DEFINE top_10_pay FIRST ---
    # Filter for the selected position level and get the top 10 averages
    top_10_pay = df_original[df_original['positionLevels'] == user_pos].groupby('postedCompany_name')['average_salary'].mean().nlargest(10).reset_index()
    
    # --- STEP 2: CLEAN AND WRAP NAMES ---
    def clean_and_wrap(name):
        # Remove PTE LTD variations (case insensitive)
        clean_name = re.sub(r'(?i)\s+PTE\.?\s+LTD\.?', '', str(name)).strip()
        # Wrap at 16 chars using <br> for Plotly
        return "<br>".join(textwrap.wrap(clean_name, width=16))

    # Apply the function to create a new 'Company' column
    top_10_pay['Company'] = top_10_pay['postedCompany_name'].apply(clean_and_wrap)
    
    # --- STEP 3: CREATE THE CHART ---
    fig_top_pay = px.bar(
        top_10_pay, 
        x='average_salary', 
        y='Company', 
        orientation='h', 
        color='average_salary', 
        color_continuous_scale='Viridis',
        labels={'average_salary': 'Average<br>Salary', 'Company': ''} 
    )
    
    # --- STEP 4: CONFIGURE LAYOUT ---
    fig_top_pay.update_layout(
        height=550, # More height prevents overlap of wrapped text
        margin=dict(l=140, r=20, t=20, b=50), 
        yaxis={
            'categoryorder': 'total ascending', 
            'title': None,
            'automargin': True # Key to making wrapped labels fit
        },
        xaxis_title="Average Salary ($)",
        coloraxis_colorbar=dict(title="Average<br>Salary")
    )

    # Adjust font size for better fit
    fig_top_pay.update_yaxes(tickfont=dict(size=11))
    
    st.plotly_chart(fig_top_pay, use_container_width=True)

with row1_col3:
    st.subheader("3. Salary Distribution (Violin View)")

    # 1. Filter data to remove 0s and extreme outliers that squish the plot
    # Setting a cap at 30k makes the 'violin' shape actually visible
    v_data = df_original[(df_original['average_salary'] > 0) & (df_original['average_salary'] < 30000)]

    # 2. Setup the plot
    sns.set_theme(style="whitegrid")
    fig_v, ax_v = plt.subplots(figsize=(10, 8)) # Increased height for better shape

    # 3. Explicitly set hue=x to avoid the palette warning and get the shapes back
    sns.violinplot(
        data=v_data, 
        x="positionLevels", 
        y="average_salary", 
        hue="positionLevels", # This ensures colors apply correctly
        palette="bright", 
        ax=ax_v,
        legend=False # Removes redundant legend
    )

    ax_v.set_title("Salary Density (Up to $30k)")
    ax_v.set_ylabel("Monthly Salary ($)")
    plt.xticks(rotation=45)

    st.pyplot(fig_v)


# ---------------- 5. ROW 2: TWO COLUMNS (SUNBURSTS) ---------------- #
st.divider()
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    # --- CHART 4: TOP 10 CATEGORIES ---
    if not df_exploded.empty:
        st.subheader("4. Top 10 Job Categories: Viewership & Salary")
        
        # Define data to avoid NameError
        top_cats = df_exploded['category_name'].value_counts().nlargest(10).index
        df_filtered = df_exploded[df_exploded['category_name'].isin(top_cats)].copy()
        
        # Ensure salary is numeric
        df_filtered['salary_maximum'] = pd.to_numeric(df_filtered['salary_maximum'], errors='coerce').fillna(0)

        fig1 = px.sunburst(
            df_filtered, 
            path=['category_name', 'status_jobStatus'], 
            values='metadata_totalNumberOfView', 
            color='salary_maximum', 
            # Using RdBu_r (reversed) puts 'Higher Salary' in Blue and 'Lower' in Red
            color_continuous_scale='RdBu', 
            # FIX: Providing [min, max] values to avoid SyntaxError
            # This range forces dramatic color shifts between 3k and 9k
            range_color=[3000, 9000], 
            color_continuous_midpoint=6000,
            labels={'salary_maximum': 'Max Salary'}
        )
        
        # Format the legend to match your 0-12k requirement
        fig1.update_coloraxes(
            colorbar_tickvals=[0, 3000, 6000, 9000, 12000],
            colorbar_ticktext=['0k', '3k', '6k', '9k', '12k'],
            colorbar_title="Max Salary"
        )
        
        fig1.update_layout(height=600)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("The 'categories' data is empty or missing. Please check your data source.")

with row2_col2:
    # --- CHART 5: INTERACTIVE SUNBURST (FIXED & UPDATED) ---
    st.subheader("5. Interactive Sunburst Chart (Top 10 Filtered)")

    # 1. Define selectable options (exclude salary from manual selection to force it as outermost)
    selectable_options = [opt for opt in actual_full if opt not in ('status_jobStatus', 'employmentTypes', 'average_salary')]

    # FIX: Define 'path_selection' here so it is not undefined below
    path_selection = st.multiselect(
        'Select categories for hierarchy:', 
        options=selectable_options, 
        default=['postedCompany_name', 'positionLevels']
    )

    if path_selection:
        # Top 10 Filter Logic
        top_10_companies = df_original['postedCompany_name'].value_counts().nlargest(10).index
        df_c2 = df_original[df_original['postedCompany_name'].isin(top_10_companies)].copy()
        
        # 2. ENSURE SALARY IS ALWAYS OUTERMOST
        df_c2['salary_label'] = (df_c2['average_salary'] / 1000).round(0).astype(int).astype(str) + "k"
        final_path = path_selection + ['salary_label']
        
        # 3. GRADIENT: 0k to 12k with 'RdBu' style
        fig2 = px.sunburst(
            df_c2, 
            path=final_path, 
            values='average_salary', 
            color='average_salary',
            color_continuous_scale='RdBu', 
            range_color=[0, 12000],  # Clips scale at 12k
            maxdepth=4
        )

        # 4. LEGEND: Update labels to show 'k' suffix
        fig2.update_coloraxes(
            colorbar_tickvals=[0, 3000, 6000, 9000, 12000],
            colorbar_ticktext=['0k', '3k', '6k', '9k', '12k'], 
            colorbar_title='Avg Salary'
        )

        fig2.update_traces(textfont_size=14, insidetextorientation='horizontal')
        fig2.update_layout(height=700, margin=dict(t=40, l=20, r=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Select at least one category to build the inner rings.")

# ---------------- 6. ROWS 3-5: FULL WIDTH ---------------- #
st.divider()
# --- CHART 6: SALARY DISTRIBUTION ---
st.subheader("6. Salary Distribution")
sel_col = st.selectbox('Color by (Safe Options):', options=[o for o in safe_options if o != 'average_salary'])
fig3 = px.histogram(df_original, x='average_salary', color=sel_col, range_x=[0, 20000])
st.plotly_chart(fig3, use_container_width=True)

import plotly.graph_objects as go

# 1. Extract a Seaborn palette as a list of hex colors
# 'deep' is the classic Seaborn look, 'bright' is better for distinction
seaborn_colors = sns.color_palette("bright", 10).as_hex()

# --- CHART 7: STRATEGIC TREND ---
st.divider()
st.subheader("7. Market Momentum: Pay vs. Visibility")

df_sampled = df_original.iloc[::5] 

fig4 = px.scatter(df_sampled, 
                 x="average_salary", 
                 y="metadata_totalNumberOfView", 
                 color="positionLevels",
                 trendline="lowess", 
                 trendline_scope='overall',
                 trendline_color_override="black",
                 # Apply the Seaborn colors here:
                 color_discrete_sequence=seaborn_colors, 
                 render_mode='svg',             
                 title="Market Momentum (Seaborn Color Style)")

fig4.update_traces(marker=dict(size=7, opacity=0.5))
fig4.update_layout(height=600, template="plotly_white")

st.plotly_chart(fig4, use_container_width=True, config={'scrollZoom': True})


# --- CHART 8: SALARY FLOW CHORD DIAGRAM ---
st.divider()
st.subheader("8. Salary Flow: Position Levels vs. Job Titles")
st.write("This diagram visualizes the flow of salary value between broad levels and specific roles.")

# 1. Prepare Data
# Aggregate by mean Average Salary to find typical pay flow per level-title pair
# Limiting to top 15 titles for a cleaner, non-overlapping circular layout
top_titles_list = df_original['title'].value_counts().nlargest(15).index
df_chord_data = df_original[df_original['title'].isin(top_titles_list)]

chord_df = df_chord_data.groupby(['positionLevels', 'title'])['average_salary'].mean().reset_index()
chord_df.columns = ['source', 'target', 'value']

# 2. Build the Chord Object
hv.extension('bokeh')
nodes_list = hv.Dataset(pd.concat([chord_df['source'], chord_df['target']]).unique(), 'index')
chord_chart = hv.Chord((chord_df, nodes_list))

# 3. Apply Styling for Fair Salary Clarity
# --- UPDATED STYLING FOR CHART 5 ---
chord_chart.opts(
    opts.Chord(
        cmap='Category20',
        edge_color=hv.dim('source').str(), 
        labels='index', 
        node_color=hv.dim('index').str(),
        width=750, 
        height=750,
        label_text_font_size='10pt',
        edge_alpha=0.7,
        node_size=12,
        # ADD THIS LINE TO ENABLE POPUPS
        tools=['hover']
    )
)

# 4. Render using the new custom component
# Pass the rendered bokeh figure and a unique key to prevent conflicts
chord_rendered = hv.render(chord_chart, backend='bokeh')
streamlit_bokeh(chord_rendered, use_container_width=True, key="fair_salary_chord")
