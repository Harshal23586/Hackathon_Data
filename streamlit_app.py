import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
from io import BytesIO
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import plotly.graph_objects as go
import plotly.express as px

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
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        # Read the Excel file
        df = pd.read_excel('Events Participation Updated.xlsx')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check for date columns (case insensitive)
        date_columns = []
        for col in df.columns:
            if 'date' in str(col).lower():
                date_columns.append(col)
        
        # Try to convert date columns
        for date_col in date_columns:
            try:
                # Try multiple date formats
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                # Rename to standard 'Date' for consistency
                df = df.rename(columns={date_col: 'Date'})
                break  # Use the first successful date column
            except:
                continue
        
        # If no date column found or converted, check for timestamp strings
        if 'Date' not in df.columns:
            # Check if any column contains date-like strings
            for col in df.columns:
                sample_value = str(df[col].iloc[0]) if len(df) > 0 else ""
                if any(date_indicator in sample_value.lower() for date_indicator in ['-', '/', '2024', '2025', '2026']):
                    try:
                        df['Date'] = pd.to_datetime(df[col], errors='coerce')
                        break
                    except:
                        continue
        
        # Extract year and month from date if available
        if 'Date' in df.columns:
            # Check if Date column is datetime
            if pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month_name()
                df['Quarter'] = df['Date'].dt.quarter
                df['Day'] = df['Date'].dt.day
                df['Weekday'] = df['Date'].dt.day_name()
            else:
                # If not datetime, try to convert
                try:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df['Year'] = df['Date'].dt.year
                    df['Month'] = df['Date'].dt.month_name()
                    df['Quarter'] = df['Date'].dt.quarter
                except:
                    # If conversion fails, remove the Date column
                    df = df.drop('Date', axis=1, errors='ignore')
        
        st.success(f"✅ Successfully loaded {len(df)} records")
        
        # Show column info for debugging
        st.sidebar.info(f"Columns found: {', '.join(df.columns.tolist()[:10])}")
        if len(df.columns) > 10:
            st.sidebar.info(f"... and {len(df.columns) - 10} more columns")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        # Create sample data as fallback
        return create_sample_data()

def create_sample_data():
    """Create sample data if Excel file can't be loaded"""
    np.random.seed(42)
    n_records = 500
    
    courses = ['CSE', 'AIML', 'AI', 'DS', 'IT', 'ENTC', 'Electrical', 'Civil', 'Mechanical', 'MBA']
    events = ['SIH 2025 Top 50', 'RACKATHON', 'Sankalpana 2K25', 'GHRHack 2.0', 
              'DIPEX', 'AAVISHKAR 2K25', 'Innovate4FinLit', 'MumbaiHacks 2025',
              'Code Veda 2.0', 'AMD Slingshot', 'HackFusion', 'Kurukshetra']
    venues = ['GHRCEM, Jalgaon', 'GRUA,Amravati', 'AICTE, MoE, Govt of India',
              'Army Institute of Technology, Pune', 'MSIS', 'KBCNMU, Jalgaon',
              'Google Developer Groups', 'Sandip University']
    
    data = {
        'Sr. No.': list(range(1, n_records + 1)),
        'Candidate Name': [f'Candidate {i}' for i in range(1, n_records + 1)],
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_records, p=[0.55, 0.43, 0.02]),
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
    df['Quarter'] = df['Date'].dt.quarter
    
    return df

# Helper function for select all functionality
def select_all_options(options, key):
    col1, col2 = st.columns([4, 1])
    with col1:
        selected = st.multiselect(
            f"Select {key}",
            options=options,
            default=options,
            key=f"select_{key}"
        )
    with col2:
        if st.button("Select All", key=f"all_{key}"):
            st.session_state[f"select_{key}"] = options
            st.rerun()
        if st.button("Clear All", key=f"clear_{key}"):
            st.session_state[f"select_{key}"] = []
            st.rerun()
    return selected

# Generate PDF report
def generate_pdf_report(df, filters_applied, summary_stats):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=30
    )
    story.append(Paragraph("Event Participation Report", title_style))
    
    # Report Date
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Summary Statistics
    story.append(Paragraph("Summary Statistics", styles['Heading2']))
    summary_data = [['Metric', 'Value']]
    for metric, value in summary_stats.items():
        summary_data.append([metric, str(value)])
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Filters Applied
    if filters_applied:
        story.append(Paragraph("Filters Applied", styles['Heading2']))
        filter_data = [['Filter', 'Value']]
        for filter_name, filter_value in filters_applied.items():
            filter_data.append([filter_name, str(filter_value)])
        
        filter_table = Table(filter_data)
        filter_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10B981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(filter_table)
        story.append(Spacer(1, 20))
    
    # Data Sample
    story.append(Paragraph("Data Sample (First 50 Records)", styles['Heading2']))
    
    # Prepare data for table (limit to 50 rows for PDF)
    pdf_data = [df.columns.tolist()]
    for i, row in df.head(50).iterrows():
        pdf_data.append([str(cell)[:50] for cell in row.tolist()])  # Truncate long values
    
    data_table = Table(pdf_data, repeatRows=1)
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6B7280')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')])
    ]))
    story.append(data_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    # Load data
    df = load_data()
    
    # Main header
    st.markdown('<h1 class="main-header">🎓 Event Participation Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize session state for filters
    if 'filters' not in st.session_state:
        st.session_state.filters = {}
    
    # Sidebar with enhanced filters
    st.sidebar.title("🎯 Advanced Filters")
    st.sidebar.markdown("---")
    
    filters_applied = {}
    
    # Date Range Filter
    if 'Date' in df.columns:
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
            mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
            df = df[mask]
            filters_applied['Date Range'] = f"{start_date} to {end_date}"
    
    # Course Filter with Select All
    if 'Course' in df.columns:
        st.sidebar.subheader("🎓 Courses")
        all_courses = sorted([c for c in df['Course'].unique() if pd.notna(c)])
        selected_courses = select_all_options(all_courses, 'courses')
        
        if selected_courses:
            df = df[df['Course'].isin(selected_courses)]
            filters_applied['Courses'] = f"{len(selected_courses)} selected"
    
    # Year Filter (Academic Year)
    if 'Year' in df.columns:
        st.sidebar.subheader("📚 Academic Year")
        all_years = sorted([str(y) for y in df['Year'].unique() if pd.notna(y)])
        selected_years = select_all_options(all_years, 'years')
        
        if selected_years:
            df = df[df['Year'].astype(str).isin(selected_years)]
            filters_applied['Academic Years'] = f"{len(selected_years)} selected"
    
    # Event Filter
    if 'Event Title' in df.columns:
        st.sidebar.subheader("🎯 Events")
        all_events = sorted([str(e) for e in df['Event Title'].unique() if pd.notna(e)])
        selected_events = select_all_options(all_events, 'events')
        
        if selected_events:
            df = df[df['Event Title'].isin(selected_events)]
            filters_applied['Events'] = f"{len(selected_events)} selected"
    
    # Gender Filter
    if 'Gender' in df.columns:
        st.sidebar.subheader("⚧️ Gender")
        all_genders = sorted([str(g) for g in df['Gender'].unique() if pd.notna(g)])
        selected_genders = select_all_options(all_genders, 'genders')
        
        if selected_genders:
            df = df[df['Gender'].isin(selected_genders)]
            filters_applied['Genders'] = f"{len(selected_genders)} selected"
    
    # Venue Filter
    if 'Venue' in df.columns:
        st.sidebar.subheader("📍 Venues")
        all_venues = sorted([str(v) for v in df['Venue'].unique() if pd.notna(v)])
        selected_venues = select_all_options(all_venues, 'venues')
        
        if selected_venues:
            df = df[df['Venue'].isin(selected_venues)]
            filters_applied['Venues'] = f"{len(selected_venues)} selected"
    
    # Month Filter
    if 'Month' in df.columns:
        st.sidebar.subheader("📆 Months")
        months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        all_months = [m for m in months_order if m in df['Month'].unique()]
        selected_months = select_all_options(all_months, 'months')
        
        if selected_months:
            df = df[df['Month'].isin(selected_months)]
            filters_applied['Months'] = f"{len(selected_months)} selected"
    
    # Advanced filters section
    with st.sidebar.expander("⚙️ Advanced Filters"):
        # Registration count filter
        if 'Candidate Name' in df.columns:
            participant_counts = df['Candidate Name'].value_counts()
            min_participations, max_participations = st.slider(
                "Minimum-Maximum Participations per Person",
                int(participant_counts.min()),
                int(participant_counts.max()),
                (int(participant_counts.min()), int(participant_counts.max()))
            )
            
            active_participants = participant_counts[
                (participant_counts >= min_participations) & 
                (participant_counts <= max_participations)
            ].index
            
            df = df[df['Candidate Name'].isin(active_participants)]
            filters_applied['Participations Range'] = f"{min_participations}-{max_participations}"
    
    # Key Metrics in a more visual way
    st.markdown('<h2 class="sub-header">📊 Performance Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        participants = df['Candidate Name'].nunique() if 'Candidate Name' in df.columns else 0
        st.markdown(f'<div class="metric-value">{participants:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">👥 Total Participants</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        events = df['Event Title'].nunique() if 'Event Title' in df.columns else 0
        st.markdown(f'<div class="metric-value">{events:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">🎯 Unique Events</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        registrations = len(df)
        st.markdown(f'<div class="metric-value">{registrations:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📝 Total Registrations</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Gender' in df.columns:
            male_count = (df['Gender'] == 'Male').sum()
            female_count = (df['Gender'] == 'Female').sum()
            other_count = len(df) - male_count - female_count
            ratio = f"M:{male_count} | F:{female_count}"
            if other_count > 0:
                ratio += f" | O:{other_count}"
        else:
            ratio = "N/A"
        st.markdown(f'<div class="metric-value">{ratio}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">⚖️ Gender Distribution</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Metrics Row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("📈 Avg Events/Person", 
                 f"{(len(df)/participants):.2f}" if participants > 0 else "0")
    
    with col6:
        if 'Course' in df.columns:
            top_course = df['Course'].value_counts().idxmax() if not df.empty else "N/A"
            st.metric("🏆 Top Course", top_course)
    
    with col7:
        if 'Event Title' in df.columns:
            top_event = df['Event Title'].value_counts().idxmax() if not df.empty else "N/A"
            st.metric("🔥 Most Popular Event", top_event[:20] + "..." if len(top_event) > 20 else top_event)
    
    with col8:
        if 'Date' in df.columns and not df.empty:
            busiest_month = df['Month'].value_counts().idxmax() if 'Month' in df.columns else "N/A"
            st.metric("📅 Busiest Month", busiest_month)
    
    # Data Preview with search
    st.markdown('<h2 class="sub-header">🔍 Data Explorer</h2>', unsafe_allow_html=True)
    
    # Search bar
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input("🔎 Search across all columns:", placeholder="Type to search...")
    
    with search_col2:
        show_all = st.checkbox("Show All Columns", value=False)
    
    # Apply search
    if search_query:
        mask = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include=['object']).columns:
            mask = mask | df[col].astype(str).str.contains(search_query, case=False, na=False)
        df_display = df[mask]
    else:
        df_display = df.copy()
    
    # Select columns to display
    if not show_all:
        default_cols = ['Sr. No.', 'Candidate Name', 'Gender', 'Course', 'Year', 'Event Title', 'Date', 'Venue']
        available_cols = [col for col in default_cols if col in df_display.columns]
        if len(available_cols) < 5:
            available_cols = df_display.columns.tolist()[:8]
        df_display = df_display[available_cols]
    
    st.dataframe(df_display.head(100), use_container_width=True, height=400)
    st.caption(f"Showing {len(df_display)} of {len(df)} records")
    
    # Enhanced Visualizations
    st.markdown('<h2 class="sub-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    # Create tabs for different visualization categories
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "📊 Event Analysis", 
        "👥 Participant Insights", 
        "📅 Time Series", 
        "🎓 Course & Venue"
    ])
    
    with viz_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Events Chart
            if 'Event Title' in df.columns:
                top_events = df['Event Title'].value_counts().head(15)
                fig1 = go.Figure(data=[
                    go.Bar(
                        x=top_events.values,
                        y=top_events.index,
                        orientation='h',
                        marker_color='#3B82F6',
                        text=top_events.values,
                        textposition='auto'
                    )
                ])
                fig1.update_layout(
                    title='Top 15 Events by Participation',
                    xaxis_title='Number of Participants',
                    yaxis_title='Event',
                    height=500
                )
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Event Distribution by Month
            if 'Event Title' in df.columns and 'Month' in df.columns:
                event_month = pd.crosstab(df['Event Title'], df['Month'])
                top_5_events = df['Event Title'].value_counts().head(5).index
                event_month_top = event_month.loc[top_5_events]
                
                fig2 = go.Figure()
                for event in top_5_events:
                    fig2.add_trace(go.Scatter(
                        x=event_month_top.columns,
                        y=event_month_top.loc[event],
                        mode='lines+markers',
                        name=event[:30] + "..." if len(event) > 30 else event
                    ))
                
                fig2.update_layout(
                    title='Monthly Trend for Top 5 Events',
                    xaxis_title='Month',
                    yaxis_title='Participants',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    with viz_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender Distribution Pie
            if 'Gender' in df.columns:
                gender_counts = df['Gender'].value_counts()
                fig3 = go.Figure(data=[go.Pie(
                    labels=gender_counts.index,
                    values=gender_counts.values,
                    hole=.3,
                    marker_colors=['#3B82F6', '#EF4444', '#10B981']
                )])
                fig3.update_layout(
                    title='Gender Distribution',
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Academic Year Distribution
            if 'Year' in df.columns:
                year_counts = df['Year'].value_counts().sort_index()
                fig4 = go.Figure(data=[
                    go.Bar(
                        x=year_counts.index,
                        y=year_counts.values,
                        marker_color='#8B5CF6',
                        text=year_counts.values,
                        textposition='auto'
                    )
                ])
                fig4.update_layout(
                    title='Participation by Academic Year',
                    xaxis_title='Academic Year',
                    yaxis_title='Number of Participants',
                    height=400
                )
                st.plotly_chart(fig4, use_container_width=True)
    
    with viz_tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily Timeline
            if 'Date' in df.columns:
                daily_counts = df.groupby(df['Date'].dt.date).size()
                fig5 = go.Figure(data=[
                    go.Scatter(
                        x=daily_counts.index,
                        y=daily_counts.values,
                        mode='lines+markers',
                        line=dict(color='#F59E0B', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(245, 158, 11, 0.1)'
                    )
                ])
                fig5.update_layout(
                    title='Daily Participation Timeline',
                    xaxis_title='Date',
                    yaxis_title='Number of Participants',
                    height=400
                )
                st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            # Monthly Heatmap
            if 'Month' in df.columns and 'Year' in df.columns and 'Date' in df.columns:
                df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
                monthly_counts = df['YearMonth'].value_counts().sort_index()
                
                fig6 = go.Figure(data=[
                    go.Bar(
                        x=monthly_counts.index,
                        y=monthly_counts.values,
                        marker_color=monthly_counts.values,
                        colorscale='Viridis',
                        text=monthly_counts.values,
                        textposition='auto'
                    )
                ])
                fig6.update_layout(
                    title='Monthly Participation Heatmap',
                    xaxis_title='Month',
                    yaxis_title='Number of Participants',
                    height=400,
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig6, use_container_width=True)
    
    with viz_tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # Course Distribution
            if 'Course' in df.columns:
                course_counts = df['Course'].value_counts().head(10)
                fig7 = go.Figure(data=[
                    go.Pie(
                        labels=course_counts.index,
                        values=course_counts.values,
                        hole=.2,
                        textinfo='label+percent'
                    )
                ])
                fig7.update_layout(
                    title='Top 10 Courses by Participation',
                    height=400
                )
                st.plotly_chart(fig7, use_container_width=True)
        
        with col2:
            # Venue Distribution
            if 'Venue' in df.columns:
                venue_counts = df['Venue'].value_counts().head(10)
                fig8 = go.Figure(data=[
                    go.Bar(
                        x=venue_counts.index,
                        y=venue_counts.values,
                        marker_color='linear-gradient(#00C9FF, #92FE9D)',
                        text=venue_counts.values,
                        textposition='auto'
                    )
                ])
                fig8.update_layout(
                    title='Top 10 Venues',
                    xaxis_title='Venue',
                    yaxis_title='Number of Events',
                    height=400,
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig8, use_container_width=True)
    
    # Export Section
    st.markdown('<h2 class="sub-header">📥 Export & Reports</h2>', unsafe_allow_html=True)
    
    export_col1, export_col2, export_col3, export_col4 = st.columns(4)
    
    with export_col1:
        # Export as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="event_participation_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        # Export as Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Event Participation')
            # Add summary sheet
            summary_data = {
                'Metric': ['Total Participants', 'Total Events', 'Total Registrations', 
                          'Average Events per Person', 'Data Range'],
                'Value': [
                    df['Candidate Name'].nunique() if 'Candidate Name' in df.columns else 0,
                    df['Event Title'].nunique() if 'Event Title' in df.columns else 0,
                    len(df),
                    f"{(len(df)/df['Candidate Name'].nunique()):.2f}" if 'Candidate Name' in df.columns and df['Candidate Name'].nunique() > 0 else "0",
                    f"{df['Date'].min().date() if 'Date' in df.columns else 'N/A'} to {df['Date'].max().date() if 'Date' in df.columns else 'N/A'}"
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
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                # Prepare summary statistics
                summary_stats = {
                    'Total Participants': df['Candidate Name'].nunique() if 'Candidate Name' in df.columns else 0,
                    'Total Events': df['Event Title'].nunique() if 'Event Title' in df.columns else 0,
                    'Total Registrations': len(df),
                    'Gender Distribution': dict(df['Gender'].value_counts()) if 'Gender' in df.columns else "N/A",
                    'Top Course': df['Course'].value_counts().idxmax() if 'Course' in df.columns and not df.empty else "N/A",
                    'Top Event': df['Event Title'].value_counts().idxmax() if 'Event Title' in df.columns and not df.empty else "N/A",
                    'Date Range': f"{df['Date'].min().date() if 'Date' in df.columns else 'N/A'} to {df['Date'].max().date() if 'Date' in df.columns else 'N/A'}"
                }
                
                pdf_buffer = generate_pdf_report(df, filters_applied, summary_stats)
                
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_buffer,
                    file_name="event_participation_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    with export_col4:
        # Export visualizations as images
        if st.button("🖼️ Export Charts", use_container_width=True):
            st.info("Chart export functionality would save visualizations as PNG images")
            # Note: In a production app, you would implement chart export here
    
    # Quick Stats in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Quick Stats")
    
    if not df.empty:
        if 'Course' in df.columns:
            st.sidebar.metric("Courses", df['Course'].nunique())
        if 'Event Title' in df.columns:
            st.sidebar.metric("Events", df['Event Title'].nunique())
        if 'Venue' in df.columns:
            st.sidebar.metric("Venues", df['Venue'].nunique())
        
        # Show filter summary
        if filters_applied:
            st.sidebar.markdown("---")
            st.sidebar.subheader("✅ Active Filters")
            for filter_name, filter_value in filters_applied.items():
                st.sidebar.write(f"• **{filter_name}:** {filter_value}")
        
        # Reset filters button
        if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith('select_'):
                    st.session_state[key] = []
            st.rerun()

if __name__ == "__main__":
    main()
