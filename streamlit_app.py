import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Set page configuration
st.set_page_config(
    page_title="Event Participation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    # For demo purposes, I'll recreate the dataframe from the provided content
    # In a real scenario, you would read from the Excel file
    data = []
    
    # Parse the provided file content (simplified version)
    lines = """Your provided data content would be processed here"""
    
    # Since we have the data in the file content format, let's create a sample dataframe
    # Based on the structure provided
    
    sample_data = {
        'Sr. No.': list(range(1, 201)),
        'Candidate Name': ['Prasad Manik Darade'] * 2 + ['Shaikh Mohammad Shaarif Mohammad Raees'] * 5 + ['Shweta Ishwar Bhangale'] * 4 + ['Kanishka Prafulla Wagh'] * 2,
        'Gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female'],
        'Contact': ['9373416337', '9373416337', '7057167751', '7057167751', '7057167751', '7057167751', '7057167751', 
                   '7620996534', '7620996534', '7620996534', '7620996534', '8329494856', '8329494856'],
        'Course': ['AIML', 'AIML', 'CSE', 'CSE', 'CSE', 'CSE', 'CSE', 'CSE', 'CSE', 'CSE', 'CSE', 'IT', 'IT'],
        'Year': ['SY', 'SY', 'FY', 'FY', 'FY', 'FY', 'FY', 'SY', 'SY', 'SY', 'SY', 'SY', 'SY'],
        'Event Title': ['SIH 2025 Top 50', 'RACKATHON', 'Sankalpana 2K25', 'MumbaiHacks 2025', 'Innerve X Pune', 
                       'SIH 2025 Top 50', 'RACKATHON', 'RACKATHON', 'Ideathon 9.2', 'DIPEX', 'GHRHack 2.0',
                       'Sankalpana 2K25', 'SIH 2025 Top 50'],
        'Date': ['2025-09-30', '2026-01-31', '2025-09-26', '2025-12-06', '2025-12-29', '2025-09-30', '2026-01-31',
                '2026-01-31', '2025-12-14', '2025-12-31', '2026-02-28', '2025-09-26', '2025-09-30'],
        'Venue': ['AICTE, MoE, Govt of India', 'GRUA,Amravati', 'GHRCEM, Jalgaon', 'MSIS', 'Army Institute of Technology (AIT), Pune',
                 'AICTE, MoE, Govt of India', 'GRUA,Amravati', 'GRUA,Amravati', 'Central Depository Services (India) Limited',
                 'ABVP & Srijan Trust', 'GHRCEM, Jalgaon', 'GHRCEM, Jalgaon', 'AICTE, MoE, Govt of India']
    }
    
    # Extend with more sample data to match your structure
    for i in range(14, 201):
        sample_data['Sr. No.'].append(i)
        sample_data['Candidate Name'].append(f'Candidate {i}')
        sample_data['Gender'].append(np.random.choice(['Male', 'Female']))
        sample_data['Contact'].append(f'9{np.random.randint(100000000, 999999999)}')
        sample_data['Course'].append(np.random.choice(['CSE', 'AIML', 'AI', 'DS', 'IT', 'ENTC', 'Electrical', 'Civil']))
        sample_data['Year'].append(np.random.choice(['FY', 'SY', 'TY', 'Final Year']))
        sample_data['Event Title'].append(np.random.choice(['SIH 2025 Top 50', 'RACKATHON', 'Sankalpana 2K25', 
                                                           'GHRHack 2.0', 'DIPEX', 'AAVISHKAR 2K25', 'Innovate4FinLit']))
        sample_data['Date'].append(np.random.choice(['2025-09-26', '2025-09-30', '2025-12-06', '2025-12-31', 
                                                    '2026-01-31', '2026-02-28']))
        sample_data['Venue'].append(np.random.choice(['GHRCEM, Jalgaon', 'GRUA,Amravati', 'AICTE, MoE, Govt of India',
                                                     'MSIS', 'Army Institute of Technology (AIT), Pune']))
    
    df = pd.DataFrame(sample_data)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract year and month for filtering
    df['Year_Month'] = df['Date'].dt.strftime('%Y-%m')
    df['Month'] = df['Date'].dt.month_name()
    df['Year'] = df['Date'].dt.year
    
    return df

def main():
    # Load data
    df = load_data()
    
    # Sidebar for filters
    st.sidebar.title("🎯 Filters")
    
    # Date range filter
    st.sidebar.subheader("Date Range")
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['Date'] >= pd.Timestamp(start_date)) & 
                (df['Date'] <= pd.Timestamp(end_date))]
    
    # Course filter
    st.sidebar.subheader("Course")
    all_courses = df['Course'].unique()
    selected_courses = st.sidebar.multiselect(
        "Select Courses",
        all_courses,
        default=all_courses
    )
    
    if selected_courses:
        df = df[df['Course'].isin(selected_courses)]
    
    # Year filter
    st.sidebar.subheader("Academic Year")
    all_years = df['Year'].unique()
    selected_years = st.sidebar.multiselect(
        "Select Academic Years",
        all_years,
        default=all_years
    )
    
    if selected_years:
        df = df[df['Year'].isin(selected_years)]
    
    # Event filter
    st.sidebar.subheader("Event")
    all_events = df['Event Title'].unique()
    selected_events = st.sidebar.multiselect(
        "Select Events",
        all_events,
        default=all_events
    )
    
    if selected_events:
        df = df[df['Event Title'].isin(selected_events)]
    
    # Gender filter
    st.sidebar.subheader("Gender")
    all_genders = df['Gender'].unique()
    selected_genders = st.sidebar.multiselect(
        "Select Gender",
        all_genders,
        default=all_genders
    )
    
    if selected_genders:
        df = df[df['Gender'].isin(selected_genders)]
    
    # Venue filter
    st.sidebar.subheader("Venue")
    all_venues = df['Venue'].unique()
    selected_venues = st.sidebar.multiselect(
        "Select Venues",
        all_venues,
        default=all_venues
    )
    
    if selected_venues:
        df = df[df['Venue'].isin(selected_venues)]
    
    # Main content
    st.markdown('<h1 class="main-header">🎓 Event Participation Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Participants", df['Candidate Name'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Events", df['Event Title'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Registrations", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        gender_ratio = df['Gender'].value_counts()
        male_count = gender_ratio.get('Male', 0)
        female_count = gender_ratio.get('Female', 0)
        total = male_count + female_count
        ratio = f"{male_count/total*100:.1f}% : {female_count/total*100:.1f}%" if total > 0 else "N/A"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Gender Ratio (M:F)", ratio)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data preview
    st.markdown('<h2 class="sub-header">📋 Data Preview</h2>', unsafe_allow_html=True)
    st.dataframe(df[['Candidate Name', 'Gender', 'Course', 'Year', 'Event Title', 'Date', 'Venue']].head(20))
    
    # Visualization section
    st.markdown('<h2 class="sub-header">📊 Visualizations</h2>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Event Distribution", 
        "👥 Participant Analysis", 
        "📅 Timeline View", 
        "🎓 Course-wise Analysis",
        "📍 Venue Analysis"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Event popularity chart
            event_counts = df['Event Title'].value_counts().reset_index()
            event_counts.columns = ['Event Title', 'Count']
            
            fig1 = px.bar(
                event_counts.head(10),
                x='Event Title',
                y='Count',
                title='Top 10 Most Popular Events',
                color='Count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Events by month
            monthly_events = df.groupby('Month').size().reset_index(name='Count')
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_events['Month'] = pd.Categorical(monthly_events['Month'], categories=month_order, ordered=True)
            monthly_events = monthly_events.sort_values('Month')
            
            fig2 = px.line(
                monthly_events,
                x='Month',
                y='Count',
                title='Events Distribution by Month',
                markers=True
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            gender_counts = df['Gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            
            fig3 = px.pie(
                gender_counts,
                values='Count',
                names='Gender',
                title='Gender Distribution',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Academic year distribution
            year_counts = df['Year'].value_counts().reset_index()
            year_counts.columns = ['Academic Year', 'Count']
            
            fig4 = px.bar(
                year_counts,
                x='Academic Year',
                y='Count',
                title='Participation by Academic Year',
                color='Count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        # Timeline of events
        timeline_data = df.groupby('Date').size().reset_index(name='Count')
        timeline_data = timeline_data.sort_values('Date')
        
        fig5 = px.scatter(
            timeline_data,
            x='Date',
            y='Count',
            size='Count',
            title='Event Timeline - Participation Over Time',
            color='Count',
            size_max=50
        )
        fig5.update_traces(mode='markers+lines')
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # Course-wise participation
            course_counts = df['Course'].value_counts().reset_index()
            course_counts.columns = ['Course', 'Count']
            
            fig6 = px.bar(
                course_counts,
                x='Course',
                y='Count',
                title='Participation by Course',
                color='Count',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            # Course and year heatmap
            course_year_data = df.groupby(['Course', 'Year']).size().reset_index(name='Count')
            
            fig7 = px.density_heatmap(
                course_year_data,
                x='Course',
                y='Year',
                z='Count',
                title='Course vs Academic Year Heatmap',
                color_continuous_scale='YlOrRd'
            )
            st.plotly_chart(fig7, use_container_width=True)
    
    with tab5:
        # Venue analysis
        venue_counts = df['Venue'].value_counts().reset_index()
        venue_counts.columns = ['Venue', 'Count']
        
        fig8 = px.treemap(
            venue_counts,
            path=['Venue'],
            values='Count',
            title='Event Distribution by Venue',
            color='Count',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig8, use_container_width=True)
    
    # Detailed report section
    st.markdown('<h2 class="sub-header">📄 Detailed Report</h2>', unsafe_allow_html=True)
    
    # Show filtered data
    st.write(f"Showing {len(df)} records after applying filters")
    
    # Export options
    st.markdown("---")
    st.markdown("### 📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="event_participation_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export as Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Event Participation')
        
        st.download_button(
            label="📥 Download as Excel",
            data=output.getvalue(),
            file_name="event_participation_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        # Export summary statistics
        if st.button("📊 Generate Summary Report"):
            summary_data = {
                'Metric': [
                    'Total Unique Participants',
                    'Total Events',
                    'Total Registrations',
                    'Gender Ratio (Male:Female)',
                    'Most Popular Event',
                    'Most Active Course',
                    'Most Active Month',
                    'Most Frequent Venue'
                ],
                'Value': [
                    df['Candidate Name'].nunique(),
                    df['Event Title'].nunique(),
                    len(df),
                    f"{df['Gender'].value_counts().get('Male', 0)} : {df['Gender'].value_counts().get('Female', 0)}",
                    df['Event Title'].value_counts().idxmax() if not df.empty else 'N/A',
                    df['Course'].value_counts().idxmax() if not df.empty else 'N/A',
                    df['Month'].value_counts().idxmax() if not df.empty else 'N/A',
                    df['Venue'].value_counts().idxmax() if not df.empty else 'N/A'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
            
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
    
    search_term = st.text_input("Search by name or contact:")
    
    if search_term:
        search_results = df[df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]
        st.write(f"Found {len(search_results)} results")
        st.dataframe(search_results[['Candidate Name', 'Gender', 'Course', 'Year', 'Event Title', 'Date', 'Venue']])
    
    # Statistics section
    st.markdown("---")
    st.markdown("### 📊 Advanced Statistics")
    
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Events per Participant", 
                     f"{len(df)/df['Candidate Name'].nunique():.2f}")
        
        with col2:
            most_active = df['Candidate Name'].value_counts().idxmax()
            st.metric("Most Active Participant", most_active)
        
        with col3:
            busiest_day = df['Date'].value_counts().idxmax()
            st.metric("Busiest Event Day", busiest_day.strftime('%Y-%m-%d'))

if __name__ == "__main__":
    main()
