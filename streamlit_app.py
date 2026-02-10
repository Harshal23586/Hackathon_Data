import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
from io import BytesIO
import base64

# Try to import Plotly with fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Using simpler visualizations.")

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
</style>
""", unsafe_allow_html=True)

# Helper function for select all functionality
def select_all_multiselect(label, options, default=None, key=None):
    """Create a multiselect with Select All/Clear All buttons"""
    if default is None:
        default = []
    
    # Create container
    container = st.container()
    
    # Initialize session state if not exists
    if f"{key}_selected" not in st.session_state:
        st.session_state[f"{key}_selected"] = default
    
    # Get current selection from session state
    current_selection = st.session_state[f"{key}_selected"]
    
    # Buttons for select all/clear all
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Select All", key=f"{key}_all", use_container_width=True):
            st.session_state[f"{key}_selected"] = options
            st.rerun()
    with col2:
        if st.button("✗ Clear All", key=f"{key}_clear", use_container_width=True):
            st.session_state[f"{key}_selected"] = []
            st.rerun()
    
    # Multiselect widget
    selected = container.multiselect(
        label,
        options=options,
        default=st.session_state[f"{key}_selected"],
        key=key
    )
    
    # Update session state
    st.session_state[f"{key}_selected"] = selected
    
    return selected

# Load data with duplicate column handling
@st.cache_data
def load_data():
    try:
        # Read the Excel file
        df = pd.read_excel('Events Participation Updated.xlsx')
        
        # FIX: Handle duplicate column names by renaming duplicates
        cols = pd.Series(df.columns)
        
        # Identify duplicate column names
        duplicate_mask = cols.duplicated(keep='first')
        
        if duplicate_mask.any():
            st.warning(f"⚠️ Found {duplicate_mask.sum()} duplicate column names. Renaming duplicates...")
            
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
            st.success(f"✅ Renamed duplicate columns. New columns: {', '.join(new_cols[:10])}...")
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Check for date columns and convert (handle multiple date columns)
        date_columns = []
        for col in df.columns:
            if 'date' in str(col).lower():
                date_columns.append(col)
        
        if date_columns:
            st.info(f"Found date columns: {date_columns}")
            
            # Try each date column
            for date_col in date_columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    # Rename the first successful date column to 'Date'
                    if 'Date' not in df.columns:
                        df = df.rename(columns={date_col: 'Date'})
                        st.success(f"✅ Using '{date_col}' as Date column")
                        break
                except Exception as e:
                    st.warning(f"Could not convert {date_col}: {e}")
        
        # If date conversion was successful, extract date parts
        if 'Date' in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df['Date']):
                    df['Year'] = df['Date'].dt.year
                    df['Month'] = df['Date'].dt.month_name()
                    df['Quarter'] = df['Date'].dt.quarter
                else:
                    # Try to convert
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    if pd.api.types.is_datetime64_any_dtype(df['Date']):
                        df['Year'] = df['Date'].dt.year
                        df['Month'] = df['Date'].dt.month_name()
                        df['Quarter'] = df['Date'].dt.quarter
            except Exception as e:
                st.warning(f"Could not extract date parts: {e}")
        
        # Show column info for debugging
        st.sidebar.info(f"📊 Loaded {len(df)} records")
        st.sidebar.info(f"📋 Columns: {len(df.columns)}")
        
        # Show first few column names
        col_preview = df.columns.tolist()[:10]
        if len(df.columns) > 10:
            col_preview.append(f"... and {len(df.columns) - 10} more")
        st.sidebar.info(f"🔍 Columns found: {', '.join(col_preview)}")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        # Create sample data
        return create_sample_data()

def create_sample_data():
    """Create sample event participation data"""
    np.random.seed(42)
    n_records = 200
    
    courses = ['CSE', 'AIML', 'AI', 'DS', 'IT', 'ENTC', 'Electrical', 'Civil']
    events = ['SIH 2025 Top 50', 'RACKATHON', 'Sankalpana 2K25', 'GHRHack 2.0', 'DIPEX']
    venues = ['GHRCEM, Jalgaon', 'GRUA,Amravati', 'AICTE, MoE, Govt of India']
    
    data = {
        'Sr. No.': list(range(1, n_records + 1)),
        'Candidate Name': [f'Candidate {i}' for i in range(1, n_records + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_records, p=[0.6, 0.4]),
        'Contact': [f'9{np.random.randint(100000000, 999999999):09d}' for _ in range(n_records)],
        'Course': np.random.choice(courses, n_records),
        'Year': np.random.choice(['FY', 'SY', 'TY', 'Final Year'], n_records),
        'Event Title': np.random.choice(events, n_records),
        'Date': pd.date_range('2024-01-01', periods=n_records, freq='D').tolist(),
        'Venue': np.random.choice(venues, n_records)
    }
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Quarter'] = df['Date'].dt.quarter
    
    return df

def main():
    # Load data
    df = load_data()
    
    # Debug: Show all columns
    if st.sidebar.button("🔍 Show All Columns"):
        st.sidebar.write("All Columns in DataFrame:")
        for i, col in enumerate(df.columns):
            st.sidebar.write(f"{i+1}. {col}")
    
    # Main header
    st.markdown('<h1 class="main-header">🎓 Event Participation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar with enhanced filters
    st.sidebar.title("🎯 Advanced Filters")
    st.sidebar.markdown("---")
    
    filters_applied = {}
    
    # Date Range Filter - Only if Date column exists and is datetime
    if 'Date' in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df['Date']):
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
            else:
                st.sidebar.warning("⚠️ Date column is not in datetime format")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Date filter error: {e}")
    
    # Course Filter with Select All
    course_col = None
    for col in df.columns:
        if 'course' in str(col).lower():
            course_col = col
            break
    
    if course_col:
        st.sidebar.subheader("🎓 Courses")
        all_courses = sorted([str(c) for c in df[course_col].dropna().unique() if pd.notna(c)])
        selected_courses = select_all_multiselect(
            "Select Courses",
            all_courses,
            default=all_courses[:min(3, len(all_courses))],
            key="courses"
        )
        
        if selected_courses:
            df = df[df[course_col].isin(selected_courses)]
            filters_applied['Courses'] = f"{len(selected_courses)} selected"
    
    # Event Filter with Select All
    event_col = None
    for col in df.columns:
        if 'event' in str(col).lower():
            event_col = col
            break
    
    if event_col:
        st.sidebar.subheader("🎯 Events")
        all_events = sorted([str(e) for e in df[event_col].dropna().unique() if pd.notna(e)])
        selected_events = select_all_multiselect(
            "Select Events",
            all_events,
            default=all_events[:min(3, len(all_events))],
            key="events"
        )
        
        if selected_events:
            df = df[df[event_col].isin(selected_events)]
            filters_applied['Events'] = f"{len(selected_events)} selected"
    
    # Gender Filter with Select All
    gender_col = None
    for col in df.columns:
        if 'gender' in str(col).lower():
            gender_col = col
            break
    
    if gender_col:
        st.sidebar.subheader("⚧️ Gender")
        all_genders = sorted([str(g) for g in df[gender_col].dropna().unique() if pd.notna(g)])
        selected_genders = select_all_multiselect(
            "Select Gender",
            all_genders,
            default=all_genders,
            key="genders"
        )
        
        if selected_genders:
            df = df[df[gender_col].isin(selected_genders)]
            filters_applied['Genders'] = f"{len(selected_genders)} selected"
    
    # Venue Filter with Select All
    venue_col = None
    for col in df.columns:
        if 'venue' in str(col).lower():
            venue_col = col
            break
    
    if venue_col:
        st.sidebar.subheader("📍 Venues")
        all_venues = sorted([str(v) for v in df[venue_col].dropna().unique() if pd.notna(v)])
        selected_venues = select_all_multiselect(
            "Select Venues",
            all_venues,
            default=all_venues[:min(3, len(all_venues))],
            key="venues"
        )
        
        if selected_venues:
            df = df[df[venue_col].isin(selected_venues)]
            filters_applied['Venues'] = f"{len(selected_venues)} selected"
    
    # Year Filter (Academic Year) with Select All
    year_col = None
    # First check for academic year
    for col in df.columns:
        if col in ['Year', 'Academic Year', 'Academic_Year', 'year']:
            year_col = col
            break
    
    if year_col:
        st.sidebar.subheader("📚 Academic Year")
        all_years = sorted([str(y) for y in df[year_col].dropna().unique() if pd.notna(y)])
        selected_years = select_all_multiselect(
            "Select Years",
            all_years,
            default=all_years,
            key="years"
        )
        
        if selected_years:
            df = df[df[year_col].astype(str).isin(selected_years)]
            filters_applied['Academic Years'] = f"{len(selected_years)} selected"
    
    # Key Metrics
    st.markdown('<h2 class="sub-header">📊 Performance Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Find candidate name column
    candidate_col = None
    for col in df.columns:
        if 'name' in str(col).lower() or 'candidate' in str(col).lower():
            candidate_col = col
            break
    
    with col1:
        participants = df[candidate_col].nunique() if candidate_col else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{participants:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">👥 Total Participants</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        events = df[event_col].nunique() if event_col else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{events:,}</div>', unsafe_allow_html=True)
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
            ratio = f"M:{male_count} F:{female_count}"
        else:
            ratio = "N/A"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{ratio}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">⚖️ Gender Distribution</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Preview
    st.markdown('<h2 class="sub-header">📋 Data Explorer</h2>', unsafe_allow_html=True)
    
    # Search bar
    search_query = st.text_input("🔎 Search across all columns:", placeholder="Type to search...")
    
    # Apply search
    if search_query:
        mask = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include=['object']).columns:
            mask = mask | df[col].astype(str).str.contains(search_query, case=False, na=False)
        df_display = df[mask]
    else:
        df_display = df.copy()
    
    # Display data - show all columns but limit rows
    st.dataframe(df_display.head(50), use_container_width=True, height=400)
    st.caption(f"Showing {min(50, len(df_display))} of {len(df_display)} records")
    
    # Enhanced Visualizations
    st.markdown('<h2 class="sub-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    # Create tabs for visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Event Analysis", "👥 Participant Insights", "🎓 Course & Venue"])
    
    with viz_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Events Chart
            if event_col:
                top_events = df[event_col].value_counts().head(10)
                if PLOTLY_AVAILABLE and not top_events.empty:
                    try:
                        fig1 = go.Figure(data=[
                            go.Bar(
                                x=top_events.values,
                                y=top_events.index,
                                orientation='h',
                                marker_color='#3B82F6'
                            )
                        ])
                        fig1.update_layout(
                            title='Top 10 Events by Participation',
                            xaxis_title='Number of Participants',
                            yaxis_title='Event',
                            height=400
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    except Exception as e:
                        st.error(f"Plotly error: {e}")
                        st.bar_chart(top_events)
                else:
                    st.bar_chart(top_events)
        
        with col2:
            # Event Distribution Table
            if event_col:
                event_summary = df[event_col].value_counts().reset_index()
                event_summary.columns = ['Event Title', 'Count']
                st.dataframe(event_summary, use_container_width=True, height=400)
    
    with viz_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender Distribution
            if gender_col:
                gender_counts = df[gender_col].value_counts()
                if PLOTLY_AVAILABLE and not gender_counts.empty:
                    try:
                        fig2 = go.Figure(data=[go.Pie(
                            labels=gender_counts.index,
                            values=gender_counts.values,
                            hole=.3
                        )])
                        fig2.update_layout(
                            title='Gender Distribution',
                            height=400
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    except:
                        st.bar_chart(gender_counts)
                else:
                    st.bar_chart(gender_counts)
        
        with col2:
            # Academic Year Distribution
            if year_col:
                year_counts = df[year_col].value_counts()
                if not year_counts.empty:
                    st.bar_chart(year_counts)
    
    with viz_tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Course-wise participation
            if course_col:
                course_counts = df[course_col].value_counts()
                if not course_counts.empty:
                    st.bar_chart(course_counts)
        
        with col2:
            # Venue Distribution
            if venue_col:
                venue_counts = df[venue_col].value_counts().head(10)
                if not venue_counts.empty:
                    if PLOTLY_AVAILABLE:
                        try:
                            fig3 = go.Figure(data=[
                                go.Bar(
                                    x=venue_counts.index.astype(str),
                                    y=venue_counts.values,
                                    marker_color='#10B981',
                                    text=venue_counts.values.astype(str).tolist(),
                                    textposition='auto'
                                )
                            ])
                            fig3.update_layout(
                                title='Top 10 Venues',
                                xaxis_title='Venue',
                                yaxis_title='Number of Events',
                                height=400,
                                xaxis_tickangle=45
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        except Exception as e:
                            st.error(f"Plotly error: {e}")
                            st.bar_chart(venue_counts)
                    else:
                        st.bar_chart(venue_counts)
    
    # Export Section
    st.markdown('<h2 class="sub-header">📥 Export Data</h2>', unsafe_allow_html=True)
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # Export as CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="event_participation_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        # Export as Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Event Participation')
        
        st.download_button(
            label="📊 Download Excel",
            data=output.getvalue(),
            file_name="event_participation_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Reset filters button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.endswith('_selected'):
                st.session_state[key] = []
        st.rerun()

if __name__ == "__main__":
    main()
