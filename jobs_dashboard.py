import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Jobs Market Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Jobs Market Analysis - Consulting Dashboard")
st.markdown("---")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv('sample.csv', on_bad_lines='skip')
    
    # Clean categories column
    def extract_category(cat_string):
        try:
            if pd.isna(cat_string):
                return None
            cat_data = json.loads(cat_string)
            if isinstance(cat_data, list) and len(cat_data) > 0:
                return cat_data[0].get('category', None)
            elif isinstance(cat_data, dict):
                return cat_data.get('category', None)
        except:
            return None
    
    df['categories'] = df['categories'].apply(extract_category)
    
    # Convert dates
    df['posting_year'] = pd.to_datetime(df['metadata_originalPostingDate'], errors='coerce').dt.year
    df['posting_date'] = pd.to_datetime(df['metadata_originalPostingDate'], errors='coerce')
    
    # Clean salary data
    df['average_salary'] = pd.to_numeric(df['average_salary'], errors='coerce')
    df['salary_minimum'] = pd.to_numeric(df['salary_minimum'], errors='coerce')
    df['salary_maximum'] = pd.to_numeric(df['salary_maximum'], errors='coerce')
    df['minimumYearsExperience'] = pd.to_numeric(df['minimumYearsExperience'], errors='coerce')
    
    # Filter out salary outliers
    df = df[(df['average_salary'].isna()) | 
            ((df['average_salary'] >= 1000) & (df['average_salary'] <= 1000000))]
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Year filter
years = sorted(df['posting_year'].dropna().unique())
if len(years) > 0:
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=years,
        default=years
    )
    df_filtered = df[df['posting_year'].isin(selected_years)]
else:
    df_filtered = df

# Category filter
categories = sorted(df_filtered['categories'].dropna().unique())
if len(categories) > 0:
    selected_categories = st.sidebar.multiselect(
        "Select Categories (optional)",
        options=categories,
        default=[]
    )
    if selected_categories:
        df_filtered = df_filtered[df_filtered['categories'].isin(selected_categories)]

# Salary range filter
salary_data = df_filtered['average_salary'].dropna()
if len(salary_data) > 0:
    min_sal, max_sal = st.sidebar.slider(
        "Salary Range ($)",
        min_value=int(salary_data.min()),
        max_value=int(salary_data.max()),
        value=(int(salary_data.min()), int(salary_data.max()))
    )
    df_filtered = df_filtered[(df_filtered['average_salary'].isna()) | 
                              ((df_filtered['average_salary'] >= min_sal) & 
                               (df_filtered['average_salary'] <= max_sal))]

# Display dataset info
st.sidebar.markdown("---")
st.sidebar.metric("Total Job Postings", f"{len(df_filtered):,}")
st.sidebar.metric("Categories", f"{df_filtered['categories'].nunique()}")
st.sidebar.metric("Companies", f"{df_filtered['postedCompany_name'].nunique()}")

# Main dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Trends", 
    "💼 Categories", 
    "💰 Salary Analysis", 
    "👔 Job Titles",
    "🏢 Companies"
])

# ============================================
# TAB 1: MARKET TRENDS
# ============================================
with tab1:
    st.header("Job Market Trends Over Time")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Jobs over time
        valid_dates = df_filtered.dropna(subset=['posting_year'])
        if len(valid_dates) > 0:
            yearly_trends = valid_dates['posting_year'].value_counts().sort_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_trends.index,
                y=yearly_trends.values,
                mode='lines+markers',
                name='Job Postings',
                line=dict(color='steelblue', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title="Job Postings by Year",
                xaxis_title="Year",
                yaxis_title="Number of Postings",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Year-over-Year Growth")
        if len(yearly_trends) > 1:
            growth_data = []
            for i in range(1, len(yearly_trends)):
                prev_year = yearly_trends.iloc[i-1]
                curr_year = yearly_trends.iloc[i]
                growth = ((curr_year - prev_year) / prev_year) * 100
                year = yearly_trends.index[i]
                growth_data.append({
                    'Year': int(year),
                    'Growth': growth,
                    'Jobs': int(curr_year)
                })
            
            growth_df = pd.DataFrame(growth_data)
            for _, row in growth_df.iterrows():
                trend = "📈" if row['Growth'] > 0 else "📉"
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.metric(str(row['Year']), f"{row['Growth']:+.1f}%")
                with col_b:
                    st.write(f"{trend} {row['Jobs']:,} jobs")
    
    # Monthly trends
    st.subheader("Monthly Posting Activity")
    monthly_data = df_filtered.dropna(subset=['posting_date'])
    if len(monthly_data) > 0:
        monthly_counts = monthly_data.groupby(monthly_data['posting_date'].dt.to_period('M')).size()
        monthly_counts.index = monthly_counts.index.to_timestamp()
        
        fig = px.line(
            x=monthly_counts.index,
            y=monthly_counts.values,
            labels={'x': 'Month', 'y': 'Job Postings'},
            title="Monthly Job Posting Volume"
        )
        fig.update_traces(line_color='coral')
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 2: CATEGORIES
# ============================================
with tab2:
    st.header("Job Categories Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Categories by Volume")
        top_categories = df_filtered['categories'].value_counts().head(15)
        
        fig = px.bar(
            x=top_categories.values,
            y=top_categories.index,
            orientation='h',
            labels={'x': 'Number of Jobs', 'y': 'Category'},
            title="Top 15 Job Categories"
        )
        fig.update_traces(marker_color='steelblue')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Category Growth Trends")
        
        # Calculate growth for top categories
        if 'posting_year' in df_filtered.columns:
            recent_years = df_filtered[df_filtered['posting_year'] >= df_filtered['posting_year'].max() - 2]
            top_10_cats = df_filtered['categories'].value_counts().head(10).index
            
            growth_analysis = []
            for category in top_10_cats:
                yearly = recent_years[recent_years['categories'] == category]['posting_year'].value_counts().sort_index()
                
                if len(yearly) >= 2:
                    first_year_jobs = yearly.iloc[0]
                    last_year_jobs = yearly.iloc[-1]
                    growth_rate = ((last_year_jobs - first_year_jobs) / first_year_jobs) * 100
                    
                    growth_analysis.append({
                        'Category': category[:30],
                        'Growth': growth_rate
                    })
            
            if growth_analysis:
                growth_df = pd.DataFrame(growth_analysis).sort_values('Growth')
                
                colors = ['green' if x > 0 else 'red' for x in growth_df['Growth']]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=growth_df['Growth'],
                    y=growth_df['Category'],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{x:+.1f}%" for x in growth_df['Growth']],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Category Growth Rate (%)",
                    xaxis_title="Growth Rate (%)",
                    yaxis_title="Category",
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Legend
                st.markdown("**Legend:** 🟢 Green = Growth | 🔴 Red = Decline")

    # Category trend over time
    st.subheader("Category Trends Over Time")
    
    # Allow user to select categories to compare
    available_cats = df_filtered['categories'].value_counts().head(10).index.tolist()
    selected_cats = st.multiselect(
        "Select categories to compare",
        options=available_cats,
        default=available_cats[:3]
    )
    
    if selected_cats and 'posting_year' in df_filtered.columns:
        fig = go.Figure()
        
        for category in selected_cats:
            cat_data = df_filtered[df_filtered['categories'] == category]
            yearly = cat_data['posting_year'].value_counts().sort_index()
            
            fig.add_trace(go.Scatter(
                x=yearly.index,
                y=yearly.values,
                mode='lines+markers',
                name=category[:30]
            ))
        
        fig.update_layout(
            title="Category Comparison Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Jobs",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3: SALARY ANALYSIS
# ============================================
with tab3:
    st.header("Salary Analysis & Benchmarks")
    
    # Overall salary stats
    salary_stats = df_filtered['average_salary'].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Median Salary", f"${salary_stats['50%']:,.0f}")
    col2.metric("Average Salary", f"${salary_stats['mean']:,.0f}")
    col3.metric("Min Salary", f"${salary_stats['min']:,.0f}")
    col4.metric("Max Salary", f"${salary_stats['max']:,.0f}")
    
    st.markdown("---")
    
    # Salary by title
    st.subheader("💼 Salary by Job Title")
    
    salary_data = df_filtered[['title', 'average_salary']].dropna()
    title_counts = salary_data['title'].value_counts()
    top_titles = title_counts[title_counts >= 10].head(20).index
    
    salary_benchmarks = []
    for title in top_titles:
        title_salaries = salary_data[salary_data['title'] == title]['average_salary']
        salary_benchmarks.append({
            'Title': title[:40],
            'Median': title_salaries.median(),
            'Q25': title_salaries.quantile(0.25),
            'Q75': title_salaries.quantile(0.75),
            'Count': len(title_salaries)
        })
    
    salary_df = pd.DataFrame(salary_benchmarks).sort_values('Median', ascending=False).head(15)
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=salary_df['Title'],
        x=salary_df['Median'],
        orientation='h',
        name='Median',
        marker_color='steelblue',
        text=[f"${x:,.0f}" for x in salary_df['Median']],
        textposition='outside'
    ))
    
    # Add error bars for Q25-Q75 range
    fig.add_trace(go.Scatter(
        y=salary_df['Title'],
        x=salary_df['Median'],
        mode='markers',
        marker=dict(size=0.1, color='rgba(0,0,0,0)'),
        error_x=dict(
            type='data',
            symmetric=False,
            array=salary_df['Q75'] - salary_df['Median'],
            arrayminus=salary_df['Median'] - salary_df['Q25'],
            color='black',
            thickness=1.5
        ),
        showlegend=False,
        hovertext=[f"25th: ${q25:,.0f}<br>Median: ${med:,.0f}<br>75th: ${q75:,.0f}" 
                   for q25, med, q75 in zip(salary_df['Q25'], salary_df['Median'], salary_df['Q75'])],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="Top 15 Job Titles by Median Salary (with 25th-75th percentile range)",
        xaxis_title="Salary ($)",
        yaxis_title="Job Title",
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Salary by category
    st.subheader("🏭 Salary by Category")
    
    category_salary = df_filtered[['categories', 'average_salary']].dropna()
    top_cats = category_salary['categories'].value_counts().head(15).index
    
    cat_sal_data = []
    for category in top_cats:
        cat_salaries = category_salary[category_salary['categories'] == category]['average_salary']
        cat_sal_data.append({
            'Category': category[:40],
            'Median': cat_salaries.median(),
            'Count': len(cat_salaries)
        })
    
    cat_sal_df = pd.DataFrame(cat_sal_data).sort_values('Median', ascending=True)
    
    fig = px.bar(
        cat_sal_df,
        x='Median',
        y='Category',
        orientation='h',
        labels={'Median': 'Median Salary ($)', 'Category': 'Job Category'},
        title="Median Salary by Category",
        text=[f"${x:,.0f}" for x in cat_sal_df['Median']]
    )
    fig.update_traces(marker_color='coral', textposition='outside')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Salary vs Experience
    st.subheader("📊 Salary vs Years of Experience")
    
    exp_salary = df_filtered[['minimumYearsExperience', 'average_salary']].dropna()
    exp_salary = exp_salary[(exp_salary['minimumYearsExperience'] >= 0) & 
                            (exp_salary['minimumYearsExperience'] <= 30)]
    
    if len(exp_salary) > 0:
        exp_groups = exp_salary.groupby('minimumYearsExperience')['average_salary'].agg(['median', 'count'])
        exp_groups = exp_groups[exp_groups['count'] >= 5].reset_index()
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=exp_groups['minimumYearsExperience'],
            y=exp_groups['median'],
            mode='markers',
            marker=dict(
                size=exp_groups['count']/5,
                color=exp_groups['median'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Median<br>Salary ($)")
            ),
            text=[f"Experience: {exp} years<br>Median: ${sal:,.0f}<br>Jobs: {cnt}" 
                  for exp, sal, cnt in zip(exp_groups['minimumYearsExperience'], 
                                          exp_groups['median'], 
                                          exp_groups['count'])],
            hoverinfo='text'
        ))
        
        # Trend line
        fig.add_trace(go.Scatter(
            x=exp_groups['minimumYearsExperience'],
            y=exp_groups['median'],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Trend'
        ))
        
        fig.update_layout(
            title="Salary vs Experience (bubble size = number of jobs)",
            xaxis_title="Minimum Years of Experience",
            yaxis_title="Median Salary ($)",
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 4: JOB TITLES
# ============================================
with tab4:
    st.header("Job Title Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Common Job Titles")
        top_titles_count = df_filtered['title'].value_counts().head(20)
        
        fig = px.bar(
            x=top_titles_count.values,
            y=[t[:40] for t in top_titles_count.index],
            orientation='h',
            labels={'x': 'Number of Postings', 'y': 'Job Title'},
            title="Top 20 Job Titles"
        )
        fig.update_traces(marker_color='lightseagreen')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Position Levels")
        if df_filtered['positionLevels'].notna().any():
            pos_levels = df_filtered['positionLevels'].value_counts().head(10)
            
            fig = px.pie(
                values=pos_levels.values,
                names=pos_levels.index,
                title="Distribution by Position Level"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Employment Types")
        if df_filtered['employmentTypes'].notna().any():
            emp_types = df_filtered['employmentTypes'].value_counts().head(10)
            
            fig = px.pie(
                values=emp_types.values,
                names=emp_types.index,
                title="Distribution by Employment Type"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 5: COMPANIES
# ============================================
with tab5:
    st.header("Top Hiring Companies")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        top_companies = df_filtered['postedCompany_name'].value_counts().head(20)
        
        fig = px.bar(
            x=top_companies.values,
            y=[c[:40] for c in top_companies.index],
            orientation='h',
            labels={'x': 'Number of Job Postings', 'y': 'Company'},
            title="Top 20 Companies by Job Volume"
        )
        fig.update_traces(marker_color='mediumpurple')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Company Statistics")
        st.metric("Total Companies", f"{df_filtered['postedCompany_name'].nunique():,}")
        
        # Average jobs per company
        avg_jobs = len(df_filtered) / df_filtered['postedCompany_name'].nunique()
        st.metric("Avg Jobs per Company", f"{avg_jobs:.1f}")
        
        # Top company stats
        top_company = top_companies.index[0]
        top_company_jobs = top_companies.iloc[0]
        
        st.markdown("---")
        st.markdown("**Top Hiring Company:**")
        st.markdown(f"**{top_company}**")
        st.metric("Job Postings", f"{top_company_jobs:,}")
        
        # Salary stats for top company
        top_company_salaries = df_filtered[
            (df_filtered['postedCompany_name'] == top_company) & 
            (df_filtered['average_salary'].notna())
        ]['average_salary']
        
        if len(top_company_salaries) > 0:
            st.metric("Median Salary", f"${top_company_salaries.median():,.0f}")

# Footer
st.markdown("---")
st.markdown("### 💼 Consultant Insights")
st.info("""
**Key Takeaways:**
- Use the **Market Trends** tab to identify peak hiring periods
- Check **Categories** tab for growing vs declining industries
- Reference **Salary Analysis** for negotiation benchmarks (25th-75th percentile)
- Review **Job Titles** for in-demand roles
- Target **Top Companies** for job search focus
""")

st.markdown("---")
st.caption("📊 Jobs Market Analysis Dashboard | Data-driven career consulting")
