import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Event Participation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Try to load the Excel file
@st.cache_data
def load_data():
    try:
        # Read the Excel file
        df = pd.read_excel('Events Participation Updated.xlsx')
        
        # Show success message
        st.sidebar.success(f"✅ Loaded {len(df)} records from Excel")
        
        # Try to find the right sheet name
        if 'Event Title' not in df.columns:
            # Try to read with specific sheet name
            try:
                df = pd.read_excel('Events Participation Updated.xlsx', sheet_name='Events Participation')
                st.sidebar.success("Found sheet: Events Participation")
            except:
                # Try to read the first sheet
                df = pd.read_excel('Events Participation Updated.xlsx', sheet_name=0)
                st.sidebar.success("Reading first sheet")
        
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        # Create sample data as fallback
        return create_sample_data()

def create_sample_data():
    """Create sample data if Excel file can't be loaded"""
    data = {
        'Candidate Name': ['Prasad Manik Darade', 'Shaikh Mohammad', 'Shweta Bhangale'],
        'Gender': ['Male', 'Male', 'Female'],
        'Course': ['AIML', 'CSE', 'CSE'],
        'Year': ['SY', 'FY', 'SY'],
        'Event Title': ['SIH 2025', 'RACKATHON', 'GHRHack 2.0'],
        'Date': ['2025-09-30', '2026-01-31', '2026-02-28'],
        'Venue': ['AICTE', 'GRUA', 'GHRCEM']
    }
    return pd.DataFrame(data)

def main():
    st.title("🎓 Event Participation Dashboard")
    
    # Load data
    df = load_data()
    
    # Show data info
    st.write(f"**Total Records:** {len(df)}")
    st.write(f"**Columns Available:** {', '.join(df.columns.tolist())}")
    
    # Show first few rows
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("📊 Basic Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Candidate Name' in df.columns:
            unique_participants = df['Candidate Name'].nunique()
            st.metric("Unique Participants", unique_participants)
    
    with col2:
        if 'Event Title' in df.columns:
            unique_events = df['Event Title'].nunique()
            st.metric("Unique Events", unique_events)
    
    with col3:
        st.metric("Total Records", len(df))
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    # Course filter
    if 'Course' in df.columns:
        courses = df['Course'].dropna().unique()
        selected_courses = st.sidebar.multiselect(
            "Select Course(s)",
            options=sorted(courses),
            default=sorted(courses)[:min(3, len(courses))]
        )
        
        if selected_courses:
            df = df[df['Course'].isin(selected_courses)]
    
    # Gender filter
    if 'Gender' in df.columns:
        genders = df['Gender'].dropna().unique()
        selected_genders = st.sidebar.multiselect(
            "Select Gender(s)",
            options=sorted(genders),
            default=sorted(genders)
        )
        
        if selected_genders:
            df = df[df['Gender'].isin(selected_genders)]
    
    # Event filter
    if 'Event Title' in df.columns:
        events = df['Event Title'].dropna().unique()
        selected_events = st.sidebar.multiselect(
            "Select Event(s)",
            options=sorted(events),
            default=sorted(events)[:min(3, len(events))]
        )
        
        if selected_events:
            df = df[df['Event Title'].isin(selected_events)]
    
    # Show filtered data
    st.subheader(f"📋 Filtered Data ({len(df)} records)")
    st.dataframe(df)
    
    # Visualizations
    st.subheader("📈 Visualizations")
    
    # Event distribution
    if 'Event Title' in df.columns:
        st.write("**Event Participation Count**")
        event_counts = df['Event Title'].value_counts().head(10)
        st.bar_chart(event_counts)
    
    # Course distribution
    if 'Course' in df.columns:
        st.write("**Course-wise Participation**")
        course_counts = df['Course'].value_counts()
        st.bar_chart(course_counts)
    
    # Export data
    st.subheader("📥 Export Data")
    
    # Export as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="event_data.csv",
        mime="text/csv"
    )
    
    # Show all unique values for reference
    st.sidebar.subheader("📊 Data Summary")
    if 'Event Title' in df.columns:
        st.sidebar.write(f"**Events:** {df['Event Title'].nunique()}")
    if 'Course' in df.columns:
        st.sidebar.write(f"**Courses:** {df['Course'].nunique()}")
    if 'Year' in df.columns:
        st.sidebar.write(f"**Years:** {df['Year'].nunique()}")
    
    # Search functionality
    st.subheader("🔍 Search")
    search_term = st.text_input("Search in all columns:")
    
    if search_term:
        # Search in all string columns
        mask = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include=['object']).columns:
            mask = mask | df[col].astype(str).str.contains(search_term, case=False, na=False)
        
        search_results = df[mask]
        st.write(f"Found {len(search_results)} results")
        st.dataframe(search_results)

if __name__ == "__main__":
    main()
