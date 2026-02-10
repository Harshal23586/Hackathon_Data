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
    .filter-section {
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# FIXED: Working Select All functionality without session state conflicts
def create_filter_with_select_all(label, options, default_selected=None, key_suffix=""):
    """
    Create a filter with Select All functionality that works properly
    """
    if default_selected is None:
        default_selected = options[:min(5, len(options))]
    
    # Create a unique key for this filter
    filter_key = f"filter_{label.replace(' ', '_').lower()}_{key_suffix}"
    
    # Store selection state separately from widget state
    selection_key = f"{filter_key}_selection"
    
    # Initialize selection state
    if selection_key not in st.session_state:
        st.session_state[selection_key] = default_selected
    
    # Create columns for buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"✓ Select All", key=f"{filter_key}_select_all_btn"):
            st.session_state[selection_key] = options
            st.rerun()
    
    with col2:
        if st.button(f"✗ Clear All", key=f"{filter_key}_clear_all_btn"):
            st.session_state[selection_key] = []
            st.rerun()
    
    # Get current selection from our session state (not widget state)
    current_selection = st.session_state[selection_key]
    
    # Create the multiselect widget with current selection
    selected = st.multiselect(
        label,
        options=options,
        default=current_selection,
        key=filter_key
    )
    
    # Update our session state with the current selection
    st.session_state[selection_key] = selected
    
    return selected

# Alternative simpler approach
def simple_select_all_filter(label, options):
    """
    Simpler approach using checkboxes for select all
    """
    # Add a "Select All" checkbox
    select_all = st.checkbox(f"Select All {label}", value=True, 
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

# Load data with duplicate column handling
@st.cache_data
def load_data():
    try:
        # Read the Excel file
        df = pd.read_excel('Events Participation Updated.xlsx')
        
        # Handle duplicate column names by renaming duplicates
        cols = pd.Series(df.columns)
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
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Try to find and convert date columns
        for col in df.columns:
            if 'date' in str(col).lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Rename to 'Date' for consistency
                    if 'Date' not in df.columns:
                        df = df.rename(columns={col: 'Date'})
                    break
                except:
                    continue
        
        # Show column info
        st.sidebar.success(f"✅ Loaded {len(df)} records")
        
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
    
    # Course Filter with SIMPLE Select All (using checkbox approach)
    if course_col:
        st.sidebar.subheader("🎓 Courses")
        all_courses = sorted([str(c) for c in df[course_col].dropna().unique() if pd.notna(c)])
        selected_courses = simple_select_all_filter("Courses", all_courses)
        
        if selected_courses:
            df = df[df[course_col].isin(selected_courses)]
            filters_applied['Courses'] = f"{len(selected_courses)} selected"
    
    # Event Filter with SIMPLE Select All
    if event_col:
        st.sidebar.subheader("🎯 Events")
        all_events = sorted([str(e) for e in df[event_col].dropna().unique() if pd.notna(e)])
        selected_events = simple_select_all_filter("Events", all_events)
        
        if selected_events:
            df = df[df[event_col].isin(selected_events)]
            filters_applied['Events'] = f"{len(selected_events)} selected"
    
    # Gender Filter - SIMPLE approach
    if gender_col:
        st.sidebar.subheader("⚧️ Gender")
        all_genders = sorted([str(g) for g in df[gender_col].dropna().unique() if pd.notna(g)])
        
        # Simple approach without complex select all
        selected_genders = st.multiselect(
            "Select Gender",
            all_genders,
            default=all_genders,
            key="gender_filter"
        )
        
        if selected_genders:
            df = df[df[gender_col].isin(selected_genders)]
            filters_applied['Genders'] = f"{len(selected_genders)} selected"
    
    # Venue Filter - SIMPLE approach
    if venue_col:
        st.sidebar.subheader("📍 Venues")
        all_venues = sorted([str(v) for v in df[venue_col].dropna().unique() if pd.notna(v)])
        selected_venues = simple_select_all_filter("Venues", all_venues)
        
        if selected_venues:
            df = df[df[venue_col].isin(selected_venues)]
            filters_applied['Venues'] = f"{len(selected_venues)} selected"
    
    # Year Filter - SIMPLE approach
    if year_col:
        st.sidebar.subheader("📚 Academic Year")
        all_years = sorted([str(y) for y in df[year_col].dropna().unique() if pd.notna(y)])
        
        # Simple approach without complex select all
        selected_years = st.multiselect(
            "Select Years",
            all_years,
            default=all_years,
            key="year_filter"
        )
        
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
            ratio = f"M:{male_count} F:{female_count}"
        else:
            ratio = "N/A"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{ratio}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">⚖️ Gender Distribution</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
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
    
    # Display first 50 rows
    st.dataframe(df_display.head(50), height=400, width='stretch')
    st.caption(f"Showing {min(50, len(df_display))} of {len(df_display)} records")
    
    # Visualizations
    st.markdown('<h2 class="sub-header">📈 Analytics</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Events", "👥 Participants", "📍 Locations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if event_col:
                top_events = df[event_col].value_counts().head(10)
                if not top_events.empty:
                    if PLOTLY_AVAILABLE:
                        try:
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=top_events.values,
                                    y=top_events.index,
                                    orientation='h',
                                    marker_color='#3B82F6'
                                )
                            ])
                            fig.update_layout(
                                title='Top 10 Events',
                                xaxis_title='Participants',
                                yaxis_title='Event',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.bar_chart(top_events)
                    else:
                        st.bar_chart(top_events)
        
        with col2:
            if event_col:
                event_counts = df[event_col].value_counts().reset_index()
                event_counts.columns = ['Event', 'Count']
                st.dataframe(event_counts, height=400, width='stretch')
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if gender_col:
                gender_counts = df[gender_col].value_counts()
                if not gender_counts.empty:
                    if PLOTLY_AVAILABLE:
                        try:
                            fig = go.Figure(data=[go.Pie(
                                labels=gender_counts.index,
                                values=gender_counts.values,
                                hole=.3
                            )])
                            fig.update_layout(
                                title='Gender Distribution',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.bar_chart(gender_counts)
                    else:
                        st.bar_chart(gender_counts)
        
        with col2:
            if candidate_col:
                top_participants = df[candidate_col].value_counts().head(10)
                if not top_participants.empty:
                    st.bar_chart(top_participants)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            if course_col:
                course_counts = df[course_col].value_counts()
                if not course_counts.empty:
                    st.bar_chart(course_counts)
        
        with col2:
            if venue_col:
                venue_counts = df[venue_col].value_counts().head(10)
                if not venue_counts.empty:
                    if PLOTLY_AVAILABLE:
                        try:
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=venue_counts.values,
                                    y=venue_counts.index,
                                    orientation='h',
                                    marker_color='#10B981'
                                )
                            ])
                            fig.update_layout(
                                title='Top 10 Venues',
                                xaxis_title='Events',
                                yaxis_title='Venue',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except:
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
            file_name="event_participation.csv",
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
            file_name="event_participation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Reset button in sidebar - SIMPLIFIED
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True, key="reset_filters"):
        # Clear the app and rerun
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    main()
