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

# Try to import ReportLab for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("ReportLab not available. PDF export will not work.")

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
    .select-all-btn {
        margin-top: 5px;
        margin-bottom: 5px;
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

# Load data
@st.cache_data
def load_data():
    try:
        # Read the Excel file
        df = pd.read_excel('Events Participation Updated.xlsx')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check for date columns and convert
        date_columns = [col for col in df.columns if 'date' in str(col).lower()]
        
        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.rename(columns={date_col: 'Date'})
                break
            except:
                continue
        
        # If date conversion was successful, extract date parts
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month_name()
            df['Quarter'] = df['Date'].dt.quarter
        else:
            # Try to create Year column from academic year if available
            if 'Year' in df.columns:
                # Check if Year contains academic years like 'FY', 'SY', etc.
                if df['Year'].astype(str).str.contains('FY|SY|TY|Final', case=False, na=False).any():
                    # Keep as is
                    pass
                else:
                    # Try to extract year numbers
                    try:
                        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                    except:
                        pass
        
        st.success(f"✅ Loaded {len(df)} records")
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
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

def generate_pdf_report(df, filters_applied, summary_stats):
    """Generate PDF report"""
    if not REPORTLAB_AVAILABLE:
        st.error("ReportLab is not installed. Cannot generate PDF.")
        return None
    
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Event Participation Report", styles['Heading1']))
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
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Data Sample
        story.append(Paragraph("Data Sample (First 20 Records)", styles['Heading2']))
        
        # Prepare data for table
        pdf_data = [df.columns.tolist()]
        for i, row in df.head(20).iterrows():
            pdf_data.append([str(cell)[:30] for cell in row.tolist()])  # Truncate long values
        
        data_table = Table(pdf_data, repeatRows=1)
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6B7280')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(data_table)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def main():
    # Load data
    df = load_data()
    
    # Main header
    st.markdown('<h1 class="main-header">🎓 Event Participation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar with enhanced filters
    st.sidebar.title("🎯 Advanced Filters")
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
    
    # Course Filter with Select All
    if 'Course' in df.columns:
        st.sidebar.subheader("🎓 Courses")
        all_courses = sorted([str(c) for c in df['Course'].dropna().unique() if pd.notna(c)])
        selected_courses = select_all_multiselect(
            "Select Courses",
            all_courses,
            default=all_courses[:min(3, len(all_courses))],
            key="courses"
        )
        
        if selected_courses:
            df = df[df['Course'].isin(selected_courses)]
            filters_applied['Courses'] = f"{len(selected_courses)} selected"
    
    # Event Filter with Select All
    if 'Event Title' in df.columns:
        st.sidebar.subheader("🎯 Events")
        all_events = sorted([str(e) for e in df['Event Title'].dropna().unique() if pd.notna(e)])
        selected_events = select_all_multiselect(
            "Select Events",
            all_events,
            default=all_events[:min(3, len(all_events))],
            key="events"
        )
        
        if selected_events:
            df = df[df['Event Title'].isin(selected_events)]
            filters_applied['Events'] = f"{len(selected_events)} selected"
    
    # Gender Filter with Select All
    if 'Gender' in df.columns:
        st.sidebar.subheader("⚧️ Gender")
        all_genders = sorted([str(g) for g in df['Gender'].dropna().unique() if pd.notna(g)])
        selected_genders = select_all_multiselect(
            "Select Gender",
            all_genders,
            default=all_genders,
            key="genders"
        )
        
        if selected_genders:
            df = df[df['Gender'].isin(selected_genders)]
            filters_applied['Genders'] = f"{len(selected_genders)} selected"
    
    # Venue Filter with Select All
    if 'Venue' in df.columns:
        st.sidebar.subheader("📍 Venues")
        all_venues = sorted([str(v) for v in df['Venue'].dropna().unique() if pd.notna(v)])
        selected_venues = select_all_multiselect(
            "Select Venues",
            all_venues,
            default=all_venues[:min(3, len(all_venues))],
            key="venues"
        )
        
        if selected_venues:
            df = df[df['Venue'].isin(selected_venues)]
            filters_applied['Venues'] = f"{len(selected_venues)} selected"
    
    # Year Filter (Academic Year) with Select All
    if 'Year' in df.columns:
        st.sidebar.subheader("📚 Academic Year")
        all_years = sorted([str(y) for y in df['Year'].dropna().unique() if pd.notna(y)])
        selected_years = select_all_multiselect(
            "Select Years",
            all_years,
            default=all_years,
            key="years"
        )
        
        if selected_years:
            df = df[df['Year'].astype(str).isin(selected_years)]
            filters_applied['Academic Years'] = f"{len(selected_years)} selected"
    
    # Key Metrics
    st.markdown('<h2 class="sub-header">📊 Performance Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        participants = df['Candidate Name'].nunique() if 'Candidate Name' in df.columns else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{participants:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">👥 Total Participants</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        events = df['Event Title'].nunique() if 'Event Title' in df.columns else 0
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
        if 'Gender' in df.columns:
            male_count = (df['Gender'] == 'Male').sum()
            female_count = (df['Gender'] == 'Female').sum()
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
    
    # Display data
    display_cols = []
    for col in ['Sr. No.', 'Candidate Name', 'Gender', 'Course', 'Year', 'Event Title', 'Date', 'Venue']:
        if col in df_display.columns:
            display_cols.append(col)
    
    if display_cols:
        st.dataframe(df_display[display_cols].head(100), use_container_width=True, height=400)
    else:
        st.dataframe(df_display.head(100), use_container_width=True, height=400)
    
    st.caption(f"Showing {min(100, len(df_display))} of {len(df_display)} records")
    
    # Enhanced Visualizations
    st.markdown('<h2 class="sub-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    # Create tabs for visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Event Analysis", "👥 Participant Insights", "🎓 Course & Venue"])
    
    with viz_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Events Chart
            if 'Event Title' in df.columns:
                top_events = df['Event Title'].value_counts().head(10)
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
                    except:
                        st.bar_chart(top_events)
                else:
                    st.bar_chart(top_events)
        
        with col2:
            # Event Distribution Table
            if 'Event Title' in df.columns:
                event_summary = df['Event Title'].value_counts().reset_index()
                event_summary.columns = ['Event Title', 'Count']
                st.dataframe(event_summary, use_container_width=True, height=400)
    
    with viz_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender Distribution
            if 'Gender' in df.columns:
                gender_counts = df['Gender'].value_counts()
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
            if 'Year' in df.columns:
                year_counts = df['Year'].value_counts()
                if not year_counts.empty:
                    st.bar_chart(year_counts)
    
    with viz_tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Course-wise participation
            if 'Course' in df.columns:
                course_counts = df['Course'].value_counts()
                if not course_counts.empty:
                    st.bar_chart(course_counts)
        
        with col2:
            # Venue Distribution - FIXED VERSION
            if 'Venue' in df.columns:
                venue_counts = df['Venue'].value_counts().head(10)
                if not venue_counts.empty:
                    if PLOTLY_AVAILABLE:
                        try:
                            # Use simple color instead of gradient
                            fig3 = go.Figure(data=[
                                go.Bar(
                                    x=venue_counts.index.astype(str),  # Ensure string type
                                    y=venue_counts.values,
                                    marker_color='#10B981',
                                    text=venue_counts.values.astype(str).tolist(),  # Convert to list of strings
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
                            st.error(f"Error creating venue chart: {str(e)}")
                            st.bar_chart(venue_counts)
                    else:
                        st.bar_chart(venue_counts)
    
    # Export Section
    st.markdown('<h2 class="sub-header">📥 Export & Reports</h2>', unsafe_allow_html=True)
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
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
            # Add summary sheet
            summary_data = {
                'Metric': ['Total Participants', 'Total Events', 'Total Registrations'],
                'Value': [
                    df['Candidate Name'].nunique() if 'Candidate Name' in df.columns else 0,
                    df['Event Title'].nunique() if 'Event Title' in df.columns else 0,
                    len(df)
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
                    'Date Range': f"{df['Date'].min().date() if 'Date' in df.columns else 'N/A'} to {df['Date'].max().date() if 'Date' in df.columns else 'N/A'}"
                }
                
                pdf_buffer = generate_pdf_report(df, filters_applied, summary_stats)
                
                if pdf_buffer:
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_buffer,
                        file_name="event_participation_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    
    # Quick Stats in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Quick Stats")
    
    if not df.empty:
        if 'Course' in df.columns:
            st.sidebar.metric("Courses", df['Course'].nunique())
        if 'Event Title' in df.columns:
            st.sidebar.metric("Events", df['Event Title'].nunique())
        
        # Reset filters button
        if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.endswith('_selected'):
                    st.session_state[key] = []
            st.rerun()

if __name__ == "__main__":
    main()
