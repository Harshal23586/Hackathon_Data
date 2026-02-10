import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64

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
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Try to load data from Excel file with fallback
@st.cache_data
def load_data():
    """Load data from Excel file or create sample data if file not found"""
    try:
        # Try to read the Excel file
        df = pd.read_excel('Events Participation Updated.xlsx', sheet_name='Events Participation')
        
        st.success(f"✅ Successfully loaded {len(df)} records from Excel file")
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Check if we have the expected columns, rename if necessary
        column_mapping = {}
        if 'Sr. No.' not in df.columns and 'Sr No' in df.columns:
            column_mapping['Sr No'] = 'Sr. No.'
        if 'Candidate Name' not in df.columns and 'CandidateName' in df.columns:
            column_mapping['CandidateName'] = 'Candidate Name'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Convert date column to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Fill missing values
        df = df.fillna('')
        
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Could not load Excel file: {e}")
        st.info("Using sample data instead. Make sure 'Events Participation Updated.xlsx' is in the same directory.")
        
        # Create sample data as fallback
        return create_sample_data()

def create_sample_data():
    """Create sample event participation data"""
    np.random.seed(42)
    
    n_records = 200
    data = {
        'Sr. No.': list(range(1, n_records + 1)),
        'Candidate Name': [f'Candidate {i}' for i in range(1, n_records + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_records, p=[0.6, 0.4]),
        'Contact': [f'9{np.random.randint(100000000, 999999999)}' for _ in range(n_records)],
        'Course': np.random.choice(['CSE', 'AIML', 'AI', 'DS', 'IT', 'ENTC', 'Electrical', 'Civil'], n_records),
        'Year': np.random.choice(['FY', 'SY', 'TY', 'Final Year'], n_records),
        'Event Title': np.random.choice([
            'SIH 2025 Top 50', 'RACKATHON', 'Sankalpana 2K25', 'GHRHack 2.0', 
            'DIPEX', 'AAVISHKAR 2K25', 'Innovate4FinLit'
        ], n_records),
        'Date': pd.date_range('2025-01-01', periods=n_records, freq='D').tolist(),
        'Venue': np.random.choice([
            'GHRCEM, Jalgaon', 'GRUA,Amravati', 'AICTE, MoE, Govt of India'
        ], n_records)
    }
    
    df = pd.DataFrame(data)
    return df

def main():
    # Load data
    df = load_data()
    
    # Main header
    st.markdown('<h1 class="main-header">🎓 Event Participation Dashboard</h1>', unsafe_allow_html=True)
    
    # Show data source info
    st.info(f"📊 Loaded **{len(df)}** records | **{df['Candidate Name'].nunique()}** unique participants | **{df['Event Title'].nunique()}** unique events")
    
    # Sidebar filters
    st.sidebar.title("🎯 Filters")
    st.sidebar.markdown("---")
    
    # Date range filter (if Date column exists)
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        st.sidebar.subheader("📅 Date Range")
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
    else:
        st.sidebar.warning("Date column not found or not in datetime format")
    
    # Course filter
    if 'Course' in df.columns:
        st.sidebar.subheader("🎓 Course")
        all_courses = sorted([str(c) for c in df['Course'].unique() if pd.notna(c)])
        selected_courses = st.sidebar.multiselect(
            "Select Courses",
            all_courses,
            default=all_courses[:min(3, len(all_courses))]
        )
        
        if selected_courses:
            df = df[df['Course'].isin(selected_courses)]
    
    # Year filter
    if 'Year' in df.columns:
        st.sidebar.subheader("📚 Academic Year")
        all_years = sorted([str(y) for y in df['Year'].unique() if pd.notna(y)])
        selected_years = st.sidebar.multiselect(
            "Select Years",
            all_years,
            default=all_years
        )
        
        if selected_years:
            df = df[df['Year'].isin(selected_years)]
    
    # Event filter
    if 'Event Title' in df.columns:
        st.sidebar.subheader("🎯 Event")
        all_events = sorted([str(e) for e in df['Event Title'].unique() if pd.notna(e)])
        selected_events = st.sidebar.multiselect(
            "Select Events",
            all_events,
            default=all_events[:min(3, len(all_events))]
        )
        
        if selected_events:
            df = df[df['Event Title'].isin(selected_events)]
    
    # Gender filter
    if 'Gender' in df.columns:
        st.sidebar.subheader("⚧️ Gender")
        all_genders = sorted([str(g) for g in df['Gender'].unique() if pd.notna(g)])
        selected_genders = st.sidebar.multiselect(
            "Select Gender",
            all_genders,
            default=all_genders
        )
        
        if selected_genders:
            df = df[df['Gender'].isin(selected_genders)]
    
    # Key Metrics
    st.markdown('<h2 class="sub-header">📊 Key Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        participants = df['Candidate Name'].nunique() if 'Candidate Name' in df.columns else 0
        st.metric("👥 Total Participants", participants)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        events = df['Event Title'].nunique() if 'Event Title' in df.columns else 0
        st.metric("🎯 Total Events", events)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📝 Total Registrations", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Gender' in df.columns:
            male_count = (df['Gender'] == 'Male').sum()
            female_count = (df['Gender'] == 'Female').sum()
            total = male_count + female_count
            if total > 0:
                ratio = f"{male_count/total*100:.1f}% : {female_count/total*100:.1f}%"
            else:
                ratio = "N/A"
        else:
            ratio = "No gender data"
        st.metric("⚖️ Gender Ratio", ratio)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Preview
    st.markdown('<h2 class="sub-header">📋 Data Preview</h2>', unsafe_allow_html=True)
    
    # Show filtered count
    st.info(f"Showing **{len(df)}** records after applying filters")
    
    # Display available columns
    display_columns = []
    for col in ['Sr. No.', 'Candidate Name', 'Gender', 'Course', 'Year', 'Event Title', 'Date', 'Venue']:
        if col in df.columns:
            display_columns.append(col)
    
    if display_columns:
        st.dataframe(
            df[display_columns],
            use_container_width=True,
            height=400
        )
    else:
        st.warning("No standard columns found in the data")
        st.dataframe(df, use_container_width=True)
    
    # Visualizations
    st.markdown('<h2 class="sub-header">📈 Visualizations</h2>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Event Distribution", 
        "👥 Participant Analysis", 
        "📅 Timeline", 
        "🎓 Course Analysis"
    ])
    
    with tab1:
        if 'Event Title' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Event popularity - Bar chart
                event_counts = df['Event Title'].value_counts().head(10)
                st.subheader("Top 10 Events by Participation")
                st.bar_chart(event_counts)
            
            with col2:
                # Event distribution table
                st.subheader("Event Distribution Summary")
                event_summary = df['Event Title'].value_counts().reset_index()
                event_summary.columns = ['Event Title', 'Count']
                st.dataframe(event_summary, use_container_width=True, height=300)
        else:
            st.warning("Event Title column not found")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Gender' in df.columns:
                # Gender distribution
                gender_counts = df['Gender'].value_counts()
                st.subheader("Gender Distribution")
                st.bar_chart(gender_counts)
            else:
                st.warning("Gender column not found")
        
        with col2:
            if 'Year' in df.columns:
                # Academic year distribution
                year_counts = df['Year'].value_counts()
                st.subheader("Academic Year Distribution")
                st.bar_chart(year_counts)
            else:
                st.warning("Year column not found")
    
    with tab3:
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            # Timeline analysis
            st.subheader("Event Timeline")
            
            # Group by date
            timeline_data = df.groupby(df['Date'].dt.date).size()
            
            # Line chart for timeline
            st.line_chart(timeline_data)
            
            # Show timeline table
            st.subheader("Daily Event Count")
            timeline_df = pd.DataFrame({
                'Date': timeline_data.index,
                'Count': timeline_data.values
            })
            st.dataframe(timeline_df.sort_values('Date', ascending=False).head(20))
        else:
            st.warning("Date column not available or not in datetime format")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Course' in df.columns:
                # Course-wise participation
                course_counts = df['Course'].value_counts()
                st.subheader("Course-wise Participation")
                st.bar_chart(course_counts)
            else:
                st.warning("Course column not found")
        
        with col2:
            if 'Course' in df.columns and 'Year' in df.columns:
                # Course vs Year data
                st.subheader("Course vs Academic Year")
                course_year_data = pd.crosstab(df['Course'], df['Year'])
                st.dataframe(course_year_data, use_container_width=True)
            else:
                st.warning("Course or Year columns not found")
    
    # Export Section
    st.markdown('<h2 class="sub-header">📥 Export Data</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="event_participation.csv",
            mime="text/csv",
            help="Download filtered data as CSV file"
        )
    
    with col2:
        # Export specific columns as CSV
        if display_columns:
            filtered_csv = df[display_columns].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered CSV",
                data=filtered_csv,
                file_name="filtered_event_data.csv",
                mime="text/csv",
                help="Download filtered view as CSV"
            )
    
    with col3:
        # Generate and download summary report
        if st.button("📊 Generate Summary Report", use_container_width=True):
            summary_metrics = []
            
            # Collect all available metrics
            if 'Candidate Name' in df.columns:
                summary_metrics.append(('Total Unique Participants', df['Candidate Name'].nunique()))
            
            if 'Event Title' in df.columns:
                summary_metrics.append(('Total Events', df['Event Title'].nunique()))
            
            summary_metrics.append(('Total Registrations', len(df)))
            
            if 'Gender' in df.columns:
                male = (df['Gender'] == 'Male').sum()
                female = (df['Gender'] == 'Female').sum()
                summary_metrics.append(('Gender Ratio (M:F)', f"{male}:{female}"))
            
            if 'Event Title' in df.columns and not df.empty:
                most_popular = df['Event Title'].value_counts().idxmax()
                summary_metrics.append(('Most Popular Event', most_popular))
            
            if 'Course' in df.columns and not df.empty:
                most_active_course = df['Course'].value_counts().idxmax()
                summary_metrics.append(('Most Active Course', most_active_course))
            
            if 'Date' in df.columns and not df.empty:
                date_range = f"{df['Date'].min().date()} to {df['Date'].max().date()}"
                summary_metrics.append(('Data Date Range', date_range))
            
            # Create summary dataframe
            summary_df = pd.DataFrame(summary_metrics, columns=['Metric', 'Value'])
            
            st.dataframe(summary_df, use_container_width=True)
            
            # Download summary
            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Summary",
                data=summary_csv,
                file_name="event_summary.csv",
                mime="text/csv"
            )
    
    # Search functionality
    st.markdown("---")
    st.markdown("### 🔍 Search Participants")
    
    search_term = st.text_input("Search by name, course, or event:", placeholder="Enter search term...")
    
    if search_term:
        search_results = pd.DataFrame()
        
        # Search across available columns
        search_columns = []
        for col in ['Candidate Name', 'Course', 'Event Title', 'Venue']:
            if col in df.columns:
                search_columns.append(col)
        
        if search_columns:
            mask = pd.Series(False, index=df.index)
            for col in search_columns:
                mask = mask | df[col].astype(str).str.contains(search_term, case=False, na=False)
            
            search_results = df[mask]
        
        if not search_results.empty:
            st.success(f"Found {len(search_results)} matching records")
            st.dataframe(search_results[display_columns] if display_columns else search_results, 
                        use_container_width=True)
        else:
            st.warning("No matching records found")
    
    # Advanced Statistics
    st.markdown("---")
    st.markdown("### 📈 Advanced Statistics")
    
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Candidate Name' in df.columns:
                avg_events = len(df) / df['Candidate Name'].nunique()
                st.metric("📊 Avg Events per Participant", f"{avg_events:.2f}")
        
        with col2:
            if 'Candidate Name' in df.columns:
                most_active = df['Candidate Name'].value_counts()
                if not most_active.empty:
                    top_participant = most_active.idxmax()
                    # Truncate long names
                    display_name = top_participant[:20] + "..." if len(top_participant) > 20 else top_participant
                    st.metric("👑 Most Active Participant", display_name)
        
        with col3:
            if 'Date' in df.columns:
                date_counts = df['Date'].value_counts()
                if not date_counts.empty:
                    busiest_day = date_counts.idxmax()
                    st.metric("📅 Busiest Day", busiest_day.strftime('%d %b %Y'))

if __name__ == "__main__":
    main()
