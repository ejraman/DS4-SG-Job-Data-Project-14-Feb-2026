import ast
import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plotly
import plotly.express as px

# 1. Load the data
# Ensure 'test.csv' is in the correct location
df = pd.read_csv('SGJobData.csv') 

# Store a reference to the original dataframe for the Streamlit multiselect section
df_original = df.copy()
path_options = ('employmentTypes', 'positionLevels','postedCompany_name','title','average_salary') 
path_hue_options = ('employmentTypes', 'positionLevels','postedCompany_name')     


# 2. Preprocess the 'categories' column: replace NaN with an empty list
df['categories'] = df['categories'].fillna('[]') 

# 2. Convert string representation of list of dicts to actual Python objects
# Use literal_eval to safely evaluate the string as a Python list
df['categories'] = df['categories'].apply(ast.literal_eval)

# 3. Explode the list so each dictionary gets its own row
df_exploded = df.explode('categories').reset_index(drop=True)

# 4. Extract the 'category' string from the dictionary
df_exploded['category_name'] = df_exploded['categories'].apply(lambda x: x['category'] if isinstance(x, dict) else None)

# 5. Remove rows with no category
df_exploded = df_exploded.dropna(subset=['category_name'])
df_exploded['metadata_totalNumberOfView'] = pd.to_numeric(df_exploded['metadata_totalNumberOfView'], errors='coerce').fillna(0)

st.header("Job Analysis Dashboard")

# Create the first Sunburst Chart (Static Example)
st.subheader("1. Sunburst Chart on Category, Status, and Title Hierarchy")

fig = px.sunburst(
    df_exploded,
    path=['category_name', 'status_jobStatus', 'title'], # Hierarchy
    values='metadata_totalNumberOfView',                    # Size of slices
    color='salary_maximum',                        # Color based on value
    hover_data=['postedCompany_name'],             # Extra info on hover
    title="Job Postings by Category and Status"
)

fig.update_traces(textinfo="label+percent entry")
st.plotly_chart(fig) # Use st.plotly_chart to display in Streamlit


# Create the Second Sunburst Chart (Interactive Multiselect)
st.subheader("2. Interactive Sunburst Chart on User-Selected Features")

# Use a selection of columns that actually exist or are highly probable
path = st.multiselect(
    'Select the categorical features path',
    options=path_options, 
    default=['employmentTypes', 'positionLevels','postedCompany_name','title','average_salary'], # Provide a default
    key="multiselect_path"
)

# Fix 2: Only call px.sunburst if the path is NOT empty
if path:
    # Fix 3 & 4: Use a consistent dataframe (df_original) and a valid value column ('metadata_totalNumberOfView' is assumed for size)
    df_sun = df_original.copy()
    
    for col in path:
        # Ensure selected columns exist before trying to use them
        if col not in df_sun.columns:
            st.warning(f"Column '{col}' not found in the original data. Please check your CSV file.")
            continue
        df_sun[col] = df_sun[col].astype(str).fillna('Missing') # Handle potential NaNs

    # Assuming 'metadata_totalNumberOfView' is a valid column for the values argument
    # If not, you may need to aggregate data first or pick another numeric column
    try:
        fig_sun = px.sunburst(
            data_frame=df_sun, 
            path=path,
            values='average_salary', # Corrected from 'Avg Salary'
            title="Interactive Sunburst Chart"
        )
        st.plotly_chart(fig_sun)
    except ValueError as e:
        st.error(f"Error generating sunburst chart: {e}. Check if 'metadata_totalNumberOfView' exists and is numeric.")
else:
    st.info("Please select at least one feature for the sunburst path.")

# Fix 1: Correct the syntax error in the tuple and provide realistic options 
# that are likely present in a typical job postings dataset (assuming standard columns exist in test.csv)
available_columns = df_original.columns.tolist() 

# Filter out non-string/non-categorical columns for better UX in a path selector
categorical_cols = [col for col in available_columns if df_original[col].dtype == 'object' or col in ['number of vacancies', 'metadata_totalNumberOfView']]

st.subheader('3. Draw histogram for Avg Salary and color by (Position, Company, title) using Plotly')
select = st.selectbox('Select the category to color',
                      ('positionLevels','postedCompany_name','title'), key="select_hist_color") # Added unique key for selectbox

fig = px.histogram(data_frame=df_original,x='average_salary',color=select)
st.plotly_chart(fig, key="hist_dynamic_color") # Added unique key

    # 4. Find the relation between total_bill and tip on time (scatter plot)
st.markdown('---')
st.write('4. Find the relation between Average Salary and Maximum Salary on time using Seaborn')

fig, ax = plt.subplots()
hue_type = st.selectbox('Select the feature to hue',path_hue_options)

sns.scatterplot(x='average_salary',y='salary_maximum',hue=hue_type,ax=ax,data=df_original)
st.pyplot(fig)
