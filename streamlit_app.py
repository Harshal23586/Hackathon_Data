import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Try to import Plotly with fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.info("📊 For enhanced visualizations, install plotly: `pip install plotly`")

# Set page configuration
st.set_page_config(
    page_title="Event Participation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def simple_select_all_filter(label, options):
    """
    Simple Select All using checkbox
    """
    select_all = st.checkbox(f"📋 Select All {label}", value=True, 
                           key=f"select_all_{label.replace(' ', '_')}")
    
    if select_all:
        selected = options
    else:
        selected = st.multiselect(
            f"Choose {label}:",
            options=options,
            default=options[:min(3, len(options))],
            key=f"multiselect_{label.replace(' ', '_')}"
        )
    
    return selected

def generate_pdf_report(df, summary_stats, filters_applied):
    """
    Generate a comprehensive PDF report
    """
    buffer = BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 1: Cover Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.5, 0.7, 'Event Participation Report', 
                fontsize=24, ha='center', va='center', fontweight='bold')
        ax.text(0.5, 0.6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                fontsize=12, ha='center', va='center')
        ax.text(0.5, 0.5, f'Total Records: {len(df)}', 
                fontsize=14, ha='center', va='center')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Summary Statistics
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.5, 0.9, 'Summary Statistics', fontsize=18, ha='center', va='center', fontweight='bold')
        
        # Create table data
        table_data = [['Metric', 'Value']]
        for key, value in summary_stats.items():
            table_data.append([key, str(value)])
        
        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Top Events Chart
        if 'Event Title' in df.columns:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            top_events = df['Event Title'].value_counts().head(10)
            if not top_events.empty:
                ax.barh(range(len(top_events)), top_events.values)
                ax.set_yticks(range(len(top_events)))
                ax.set_yticklabels(top_events.index)
                ax.set_xlabel('Number of Participants')
                ax.set_title('Top 10 Events by Participation', fontsize=16, fontweight='bold')
                ax.invert_yaxis()
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        # Page 4: Gender Distribution
        if 'Gender' in df.columns:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            gender_counts = df['Gender'].value_counts()
            if not gender_counts.empty:
                wedges, texts, autotexts = ax.pie(gender_counts.values, 
                                                  labels=gender_counts.index,
                                                  autopct='%1.1f%%',
                                                  startangle=90)
                ax.set_title('Gender Distribution', fontsize=16, fontweight='bold')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        # Page 5: Course Distribution
        if 'Course' in df.columns:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            course_counts = df['Course'].value_counts().head(10)
            if not course_counts.empty:
                ax.bar(range(len(course_counts)), course_counts.values)
                ax.set_xticks(range(len(course_counts)))
                ax.set_xticklabels(course_counts.index, rotation=45, ha='right')
                ax.set_ylabel('Number of Participants')
                ax.set_title('Top 10 Courses by Participation', fontsize=16, fontweight='bold')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
    buffer.seek(0)
    return buffer

# Load data with duplicate column handling
@st.cache_data
def load_data():
    try:
        # Read the Excel file
        df = pd.read_excel('Events Participation Updated.xlsx')
        
        # Handle duplicate column names
        cols = pd.Series(df.columns)
        duplicate_mask = cols.duplicated(keep='first')
        
        if duplicate_mask.any():
            # Create new column names for duplicates
            new_cols = []
            seen = {}
            for col in df.columns:
                if col not in seen:
                    seen[col] = 1
                    new_cols.append(col)
                else:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")
            
            df.columns = new_cols
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Try to find and convert date columns
        for col in df.columns:
            if 'date' in str(col).lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if 'Date' not in df.columns:
                        df = df.rename(columns={col: 'Date'})
                    break
                except:
                    continue
        
        # Extract date parts if Date column exists
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month_name()
            df['Month_Num'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['Weekday'] = df['Date'].dt.day_name()
            df['Quarter'] = df['Date'].dt.quarter
        
        st.sidebar.success(f"✅ Loaded {len(df)} records")
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        # Create sample data
        return create_sample_data()

def create_sample_data():
    """Create sample event participation data"""
    np.random.seed(42)
    n_records = 300
    
    courses = ['CSE', 'AIML', 'AI', 'DS', 'IT', 'ENTC', 'Electrical', 'Civil', 'Mechanical']
    events = ['SIH 2025 Top 50', 'RACKATHON', 'Sankalpana 2K25', 'GHRHack 2.0', 'DIPEX', 
              'AAVISHKAR 2K25', 'Innovate4FinLit', 'MumbaiHacks 2025', 'Code Veda 2.0']
    venues = ['GHRCEM, Jalgaon', 'GRUA,Amravati', 'AICTE, MoE, Govt of India',
              'Army Institute of Technology, Pune', 'Sandip University']
    
    data = {
        'Sr. No.': list(range(1, n_records + 1)),
        'Candidate Name': [f'Candidate {i}' for i in range(1, n_records + 1)],
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_records, p=[0.55, 0.42, 0.03]),
        'Contact': [f'9{np.random.randint(100000000, 999999999):09d}' for _ in range(n_records)],
        'Course': np.random.choice(courses, n_records),
        'Year': np.random.choice(['FY', 'SY', 'TY', 'Final Year'], n_records, p=[0.25, 0.3, 0.25, 0.2]),
        'Event Title': np.random.choice(events, n_records),
        'Date': pd.date_range('2024-01-01', periods=n_records, freq='D').tolist(),
        'Venue': np.random.choice(venues, n_records)
    }
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Month_Num'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.day_name()
    df['Quarter'] = df['Date'].dt.quarter
    
    return df

def main():
    # Load data
    df = load_data()
    
    # Main header
    st.markdown('<h1 class="main-header">🎓 Event Participation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar with filters
    st.sidebar.title("🎯 Filters")
    st.sidebar.markdown("---")
    
    filters_applied = {}
    
    # Date Range Filter
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        st.sidebar.subheader("📅 Date Range")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
            filters_applied['Date Range'] = f"{start_date} to {end_date}"
    
    # Find relevant columns
    course_col = next((col for col in df.columns if 'course' in str(col).lower()), None)
    event_col = next((col for col in df.columns if 'event' in str(col).lower() and 'title' in str(col).lower()), None)
    gender_col = next((col for col in df.columns if 'gender' in str(col).lower()), None)
    venue_col = next((col for col in df.columns if 'venue' in str(col).lower()), None)
    year_col = next((col for col in df.columns if col in ['Year', 'Academic Year', 'year']), None)
    
    # Course Filter
    if course_col:
        st.sidebar.subheader("🎓 Courses")
        all_courses = sorted([str(c) for c in df[course_col].dropna().unique() if pd.notna(c)])
        selected_courses = simple_select_all_filter("Courses", all_courses)
        
        if selected_courses:
            df = df[df[course_col].isin(selected_courses)]
            filters_applied['Courses'] = f"{len(selected_courses)} selected"
    
    # Event Filter
    if event_col:
        st.sidebar.subheader("🎯 Events")
        all_events = sorted([str(e) for e in df[event_col].dropna().unique() if pd.notna(e)])
        selected_events = simple_select_all_filter("Events", all_events)
        
        if selected_events:
            df = df[df[event_col].isin(selected_events)]
            filters_applied['Events'] = f"{len(selected_events)} selected"
    
    # Gender Filter
    if gender_col:
        st.sidebar.subheader("⚧️ Gender")
        all_genders = sorted([str(g) for g in df[gender_col].dropna().unique() if pd.notna(g)])
        selected_genders = simple_select_all_filter("Gender", all_genders)
        
        if selected_genders:
            df = df[df[gender_col].isin(selected_genders)]
            filters_applied['Genders'] = f"{len(selected_genders)} selected"
    
    # Venue Filter
    if venue_col:
        st.sidebar.subheader("📍 Venues")
        all_venues = sorted([str(v) for v in df[venue_col].dropna().unique() if pd.notna(v)])
        selected_venues = simple_select_all_filter("Venues", all_venues)
        
        if selected_venues:
            df = df[df[venue_col].isin(selected_venues)]
            filters_applied['Venues'] = f"{len(selected_venues)} selected"
    
    # Year Filter
    if year_col:
        st.sidebar.subheader("📚 Academic Year")
        all_years = sorted([str(y) for y in df[year_col].dropna().unique() if pd.notna(y)])
        selected_years = simple_select_all_filter("Academic Years", all_years)
        
        if selected_years:
            df = df[df[year_col].astype(str).isin(selected_years)]
            filters_applied['Academic Years'] = f"{len(selected_years)} selected"
    
    # Key Metrics
    st.markdown('<h2 class="sub-header">📊 Performance Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Find candidate name column
    candidate_col = next((col for col in df.columns if 'name' in str(col).lower() or 'candidate' in str(col).lower()), None)
    
    with col1:
        participants = df[candidate_col].nunique() if candidate_col else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{participants:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">👥 Total Participants</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        events_count = df[event_col].nunique() if event_col else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{events_count:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">🎯 Unique Events</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        registrations = len(df)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{registrations:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📝 Total Registrations</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        if gender_col:
            male_count = (df[gender_col] == 'Male').sum()
            female_count = (df[gender_col] == 'Female').sum()
            other_count = len(df) - male_count - female_count
            ratio = f"M:{male_count} F:{female_count}"
            if other_count > 0:
                ratio += f" O:{other_count}"
        else:
            ratio = "N/A"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{ratio}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">⚖️ Gender Distribution</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if candidate_col and participants > 0:
            avg_participations = len(df) / participants
            st.metric("📈 Avg Events per Person", f"{avg_participations:.2f}")
    
    with col6:
        if event_col and not df.empty:
            most_popular_event = df[event_col].value_counts().idxmax() if not df[event_col].value_counts().empty else "N/A"
            st.metric("🔥 Most Popular Event", most_popular_event[:15] + "..." if len(most_popular_event) > 15 else most_popular_event)
    
    with col7:
        if course_col and not df.empty:
            most_active_course = df[course_col].value_counts().idxmax() if not df[course_col].value_counts().empty else "N/A"
            st.metric("🏆 Most Active Course", most_active_course)
    
    with col8:
        if 'Date' in df.columns and not df.empty:
            busiest_month = df['Month'].value_counts().idxmax() if 'Month' in df.columns else "N/A"
            st.metric("📅 Busiest Month", busiest_month)
    
    # Data Preview
    st.markdown('<h2 class="sub-header">📋 Data Explorer</h2>', unsafe_allow_html=True)
    
    # Search functionality
    search_query = st.text_input("🔎 Search across all columns:", placeholder="Type to search...")
    
    if search_query:
        mask = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include=['object']).columns:
            mask = mask | df[col].astype(str).str.contains(search_query, case=False, na=False)
        df_display = df[mask]
    else:
        df_display = df.copy()
    
    # Display data
    display_columns = []
    for col in ['Sr. No.', 'Candidate Name', 'Gender', 'Course', 'Year', 'Event Title', 'Date', 'Venue']:
        for df_col in df_display.columns:
            if col.lower() in df_col.lower():
                display_columns.append(df_col)
                break
    
    if display_columns:
        st.dataframe(df_display[display_columns].head(50), height=400, width='stretch')
    else:
        st.dataframe(df_display.head(50), height=400, width='stretch')
    
    st.caption(f"Showing {min(50, len(df_display))} of {len(df_display)} records")
    
    # Enhanced Visualizations
    st.markdown('<h2 class="sub-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Events Analysis", "👥 Participant Insights", "📅 Time Series", "🎓 Course & Venue"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if event_col:
                # Top Events Horizontal Bar Chart
                top_events = df[event_col].value_counts().head(15)
                if not top_events.empty:
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=top_events.values,
                                y=top_events.index,
                                orientation='h',
                                marker_color='crimson',
                                text=top_events.values,
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            title='Top 15 Events by Participation',
                            xaxis_title='Number of Participants',
                            yaxis_title='Event',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.bar_chart(top_events)
        
        with col2:
            if event_col and 'Month' in df.columns:
                # Event Distribution by Month
                event_month_data = pd.crosstab(df[event_col], df['Month'])
                top_events = df[event_col].value_counts().head(5).index
                
                if PLOTLY_AVAILABLE and not event_month_data.empty:
                    fig = go.Figure()
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                    
                    for idx, event in enumerate(top_events):
                        if event in event_month_data.index:
                            fig.add_trace(go.Scatter(
                                x=event_month_data.columns,
                                y=event_month_data.loc[event],
                                mode='lines+markers',
                                name=event[:20] + "..." if len(event) > 20 else event,
                                line=dict(color=colors[idx % len(colors)], width=3)
                            ))
                    
                    fig.update_layout(
                        title='Monthly Participation for Top 5 Events',
                        xaxis_title='Month',
                        yaxis_title='Participants',
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Show table instead
                    st.write("**Event Distribution by Month:**")
                    st.dataframe(event_month_data.head(10), width='stretch')
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if gender_col:
                # Gender Distribution Donut Chart
                gender_counts = df[gender_col].value_counts()
                if not gender_counts.empty:
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[go.Pie(
                            labels=gender_counts.index,
                            values=gender_counts.values,
                            hole=.4,
                            marker_colors=['#3B82F6', '#EF4444', '#10B981', '#F59E0B']
                        )])
                        fig.update_layout(
                            title='Gender Distribution',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.bar_chart(gender_counts)
        
        with col2:
            if year_col:
                # Academic Year Distribution with different colors
                year_counts = df[year_col].value_counts().sort_index()
                if not year_counts.empty:
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=year_counts.index,
                                y=year_counts.values,
                                marker_color='#8B5CF6',
                                text=year_counts.values,
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            title='Participation by Academic Year',
                            xaxis_title='Academic Year',
                            yaxis_title='Number of Participants',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.bar_chart(year_counts)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
                # Daily Timeline with smooth curve
                daily_counts = df.groupby(df['Date'].dt.date).size()
                if not daily_counts.empty:
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[
                            go.Scatter(
                                x=daily_counts.index,
                                y=daily_counts.values,
                                mode='lines',
                                line=dict(color='#F59E0B', width=3, shape='spline'),
                                fill='tozeroy',
                                fillcolor='rgba(245, 158, 11, 0.2)',
                                name='Daily Participation'
                            )
                        ])
                        fig.update_layout(
                            title='Daily Participation Timeline',
                            xaxis_title='Date',
                            yaxis_title='Number of Participants',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.line_chart(daily_counts)
        
        with col2:
            if 'Month' in df.columns:
                # Monthly Heatmap
                monthly_counts = df['Month'].value_counts().reindex([
                    'January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December'
                ]).dropna()
                
                if not monthly_counts.empty:
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=monthly_counts.index,
                                y=monthly_counts.values,
                                marker=dict(
                                    color=monthly_counts.values,
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Participants")
                                ),
                                text=monthly_counts.values,
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            title='Monthly Participation Distribution',
                            xaxis_title='Month',
                            yaxis_title='Number of Participants',
                            height=400,
                            xaxis_tickangle=45
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.bar_chart(monthly_counts)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            if course_col:
                # Course Distribution Sunburst Chart
                course_counts = df[course_col].value_counts()
                if not course_counts.empty:
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[
                            go.Pie(
                                labels=course_counts.index,
                                values=course_counts.values,
                                hole=.3,
                                textinfo='label+percent',
                                marker_colors=px.colors.qualitative.Set3
                            )
                        ])
                        fig.update_layout(
                            title='Course Distribution',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.bar_chart(course_counts)
        
        with col2:
            if venue_col:
                # Venue Distribution Horizontal Bar Chart
                venue_counts = df[venue_col].value_counts().head(10)
                if not venue_counts.empty:
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=venue_counts.values,
                                y=venue_counts.index,
                                orientation='h',
                                marker_color='linear-gradient(rgba(0, 201, 255, 0.8), rgba(146, 254, 157, 0.8))',
                                text=venue_counts.values,
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            title='Top 10 Venues',
                            xaxis_title='Number of Events',
                            yaxis_title='Venue',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.bar_chart(venue_counts)
    
    # Export Section with PDF Report
    st.markdown('<h2 class="sub-header">📥 Export & Reports</h2>', unsafe_allow_html=True)
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # Export as CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="event_participation.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        # Export as Excel with multiple sheets
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Event Participation')
            
            # Add summary sheet
            summary_data = {
                'Metric': [
                    'Total Participants', 'Total Events', 'Total Registrations',
                    'Average Events per Person', 'Gender Ratio',
                    'Date Range', 'Most Popular Event', 'Most Active Course'
                ],
                'Value': [
                    df[candidate_col].nunique() if candidate_col else 0,
                    df[event_col].nunique() if event_col else 0,
                    len(df),
                    f"{(len(df)/df[candidate_col].nunique()):.2f}" if candidate_col and df[candidate_col].nunique() > 0 else "0",
                    f"M:{male_count} F:{female_count}" if gender_col else "N/A",
                    f"{df['Date'].min().date() if 'Date' in df.columns else 'N/A'} to {df['Date'].max().date() if 'Date' in df.columns else 'N/A'}",
                    df[event_col].value_counts().idxmax() if event_col and not df.empty else "N/A",
                    df[course_col].value_counts().idxmax() if course_col and not df.empty else "N/A"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
        
        st.download_button(
            label="📊 Download Excel",
            data=output.getvalue(),
            file_name="event_participation_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with export_col3:
        # Generate and download PDF report
        if st.button("📄 Generate PDF Report", use_container_width=True, key="generate_pdf"):
            with st.spinner("Generating PDF report..."):
                # Prepare summary statistics
                summary_stats = {
                    'Total Participants': df[candidate_col].nunique() if candidate_col else 0,
                    'Total Events': df[event_col].nunique() if event_col else 0,
                    'Total Registrations': len(df),
                    'Gender Distribution': dict(df[gender_col].value_counts()) if gender_col else "N/A",
                    'Top 5 Events': dict(df[event_col].value_counts().head(5)) if event_col else "N/A",
                    'Top 5 Courses': dict(df[course_col].value_counts().head(5)) if course_col else "N/A",
                    'Date Range': f"{df['Date'].min().date() if 'Date' in df.columns else 'N/A'} to {df['Date'].max().date() if 'Date' in df.columns else 'N/A'}"
                }
                
                pdf_buffer = generate_pdf_report(df, summary_stats, filters_applied)
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name="event_participation_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf"
                )
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True, key="reset_filters"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    main()
