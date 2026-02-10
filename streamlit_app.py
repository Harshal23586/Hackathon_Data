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
    .stDataFrame {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sample data - Replace this with your actual data loading
@st.cache_data
def load_sample_data():
    """Create sample event participation data"""
    np.random.seed(42)
    
    # Create sample data
    n_records = 200
    
    data = {
        'Sr. No.': list(range(1, n_records + 1)),
        'Candidate Name': [f'Candidate {i}' for i in range(1, n_records + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_records, p=[0.6, 0.4]),
        'Contact': [f'9{np.random.randint(100000000, 999999999)}' for _ in range(n_records)],
        'Course': np.random.choice(['CSE', 'AIML', 'AI', 'DS', 'IT', 'ENTC', 'Electrical', 'Civil', 'Mechanical'], n_records),
        'Year': np.random.choice(['FY', 'SY', 'TY', 'Final Year'], n_records, p=[0.3, 0.3, 0.2, 0.2]),
        'Event Title': np.random.choice([
            'SIH 2025 Top 50', 'RACKATHON', 'Sankalpana 2K25', 'GHRHack 2.0', 
            'DIPEX', 'AAVISHKAR 2K25', 'Innovate4FinLit', 'MumbaiHacks 2025'
        ], n_records),
        'Date': pd.date_range('2025-01-01', periods=n_records, freq='D').tolist(),
        'Venue': np.random.choice([
            'GHRCEM, Jalgaon', 'GRUA,Amravati', 'AICTE, MoE, Govt of India',
            'Army Institute of Technology (AIT), Pune', 'MSIS', 'KBCNMU, Jalgaon'
        ], n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Add some duplicate entries for realistic participation data
    duplicate_indices = np.random.choice(df.index, size=50, replace=False)
    duplicates = df.loc[duplicate_indices].copy()
    duplicates['Event Title'] = np.random.choice(df['Event Title'].unique(), 50)
    df = pd.concat([df, duplicates]).reset_index(drop=True)
    
    df['Sr. No.'] = range(1, len(df) + 1)
    
    return df

def create_download_link(df, filename="event_participation_data.xlsx"):
    """Create download link for Excel file"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Event Participation')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Download Excel File</a>'
    return href

def main():
    # Load data
    df = load_sample_data()
    
    # Main header
    st.markdown('<h1 class="main-header">🎓 Event Participation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.title("🎯 Filters")
    st.sidebar.markdown("---")
    
    # Date range filter
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
    
    # Course filter
    st.sidebar.subheader("🎓 Course")
    all_courses = sorted(df['Course'].unique())
    selected_courses = st.sidebar.multiselect(
        "Select Courses",
        all_courses,
        default=all_courses[:3] if len(all_courses) > 3 else all_courses
    )
    
    if selected_courses:
        df = df[df['Course'].isin(selected_courses)]
    
    # Year filter
    st.sidebar.subheader("📚 Academic Year")
    all_years = sorted(df['Year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Years",
        all_years,
        default=all_years
    )
    
    if selected_years:
        df = df[df['Year'].isin(selected_years)]
    
    # Event filter
    st.sidebar.subheader("🎯 Event")
    all_events = sorted(df['Event Title'].unique())
    selected_events = st.sidebar.multiselect(
        "Select Events",
        all_events,
        default=all_events[:3] if len(all_events) > 3 else all_events
    )
    
    if selected_events:
        df = df[df['Event Title'].isin(selected_events)]
    
    # Gender filter
    st.sidebar.subheader("⚧️ Gender")
    all_genders = sorted(df['Gender'].unique())
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
        st.metric("👥 Total Participants", df['Candidate Name'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 Total Events", df['Event Title'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📝 Total Registrations", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        male_count = (df['Gender'] == 'Male').sum()
        female_count = (df['Gender'] == 'Female').sum()
        total = male_count + female_count
        ratio = f"{male_count/total*100:.1f}% : {female_count/total*100:.1f}%" if total > 0 else "N/A"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚖️ Gender Ratio", ratio)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Preview
    st.markdown('<h2 class="sub-header">📋 Data Preview</h2>', unsafe_allow_html=True)
    
    # Show filtered count
    st.info(f"Showing **{len(df)}** records after applying filters")
    
    # Display data
    display_columns = ['Sr. No.', 'Candidate Name', 'Gender', 'Course', 'Year', 'Event Title', 'Date', 'Venue']
    st.dataframe(
        df[display_columns],
        use_container_width=True,
        height=400
    )
    
    # Visualizations using Streamlit's native charts
    st.markdown('<h2 class="sub-header">📈 Visualizations</h2>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Event Distribution", 
        "👥 Participant Analysis", 
        "📅 Timeline", 
        "🎓 Course Analysis"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Event popularity - Bar chart
            event_counts = df['Event Title'].value_counts().head(10)
            st.subheader("Top 10 Events by Participation")
            st.bar_chart(event_counts)
        
        with col2:
            # Event distribution - Pie chart using bar chart
            st.subheader("Event Distribution")
            event_summary = df['Event Title'].value_counts()
            st.dataframe(event_summary)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            gender_counts = df['Gender'].value_counts()
            st.subheader("Gender Distribution")
            st.bar_chart(gender_counts)
        
        with col2:
            # Academic year distribution
            year_counts = df['Year'].value_counts()
            st.subheader("Academic Year Distribution")
            st.bar_chart(year_counts)
    
    with tab3:
        # Timeline analysis
        st.subheader("Event Timeline")
        
        # Group by date
        timeline_data = df.groupby(df['Date'].dt.date).size()
        timeline_df = pd.DataFrame({
            'Date': timeline_data.index,
            'Count': timeline_data.values
        })
        
        # Line chart for timeline
        st.line_chart(timeline_data)
        
        # Show timeline table
        st.subheader("Daily Event Count")
        st.dataframe(timeline_df.sort_values('Date', ascending=False).head(20))
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # Course-wise participation
            course_counts = df['Course'].value_counts()
            st.subheader("Course-wise Participation")
            st.bar_chart(course_counts)
        
        with col2:
            # Course vs Year heatmap data
            st.subheader("Course vs Academic Year")
            course_year_data = pd.crosstab(df['Course'], df['Year'])
            st.dataframe(course_year_data)
    
    # Export Section
    st.markdown('<h2 class="sub-header">📥 Export Data</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="event_participation.csv",
            mime="text/csv",
            help="Download filtered data as CSV file"
        )
    
    with col2:
        # Export as Excel
        excel_href = create_download_link(df)
        st.markdown(excel_href, unsafe_allow_html=True)
    
    with col3:
        # Generate summary report
        if st.button("📊 Generate Summary Report", use_container_width=True):
            summary_data = {
                'Metric': [
                    'Total Unique Participants',
                    'Total Events',
                    'Total Registrations',
                    'Gender Ratio (M:F)',
                    'Most Popular Event',
                    'Most Active Course',
                    'Most Frequent Venue',
                    'Data Range (Dates)'
                ],
                'Value': [
                    df['Candidate Name'].nunique(),
                    df['Event Title'].nunique(),
                    len(df),
                    f"{(df['Gender'] == 'Male').sum()}:{(df['Gender'] == 'Female').sum()}",
                    df['Event Title'].value_counts().idxmax() if not df.empty else 'N/A',
                    df['Course'].value_counts().idxmax() if not df.empty else 'N/A',
                    df['Venue'].value_counts().idxmax() if not df.empty else 'N/A',
                    f"{df['Date'].min().date()} to {df['Date'].max().date()}" if not df.empty else 'N/A'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download summary
            csv_summary = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Summary",
                data=csv_summary,
                file_name="event_summary.csv",
                mime="text/csv"
            )
    
    # Search functionality
    st.markdown("---")
    st.markdown("### 🔍 Search Participants")
    
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_term = st.text_input("Search by name, course, or event:", placeholder="Enter search term...")
    
    with search_col2:
        search_button = st.button("Search", use_container_width=True)
    
    if search_term or search_button:
        if search_term:
            # Search across multiple columns
            search_results = df[
                df['Candidate Name'].str.contains(search_term, case=False, na=False) |
                df['Course'].str.contains(search_term, case=False, na=False) |
                df['Event Title'].str.contains(search_term, case=False, na=False) |
                df['Venue'].str.contains(search_term, case=False, na=False)
            ]
        else:
            search_results = pd.DataFrame()  # Empty dataframe if no search term
        
        if not search_results.empty:
            st.success(f"Found {len(search_results)} matching records")
            st.dataframe(search_results[display_columns], use_container_width=True)
        else:
            st.warning("No matching records found")
    
    # Statistics at the bottom
    st.markdown("---")
    st.markdown("### 📈 Advanced Statistics")
    
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_events = len(df) / df['Candidate Name'].nunique()
            st.metric("📊 Avg Events per Participant", f"{avg_events:.2f}")
        
        with col2:
            most_active = df['Candidate Name'].value_counts().idxmax()
            st.metric("👑 Most Active Participant", most_active.split()[0] + " " + most_active.split()[1][0])
        
        with col3:
            busiest_day = df['Date'].value_counts().idxmax()
            st.metric("📅 Busiest Day", busiest_day.strftime('%d %b %Y'))

if __name__ == "__main__":
    main()
