@st.cache_data
def load_sample_data():
    """Load data from your Excel file"""
    try:
        df = pd.read_excel('Events Participation Updated.xlsx', sheet_name='Events Participation')
        
        # Rename columns if they have different names
        column_mapping = {
            'Sr. No.': 'Sr. No.',
            'Candidate Name': 'Candidate Name', 
            'Gender': 'Gender',
            'Contact': 'Contact',
            'Course': 'Course',
            'Year': 'Year',
            'Event Title': 'Event Title',
            'Date': 'Date',
            'Venue': 'Venue'
        }
        
        # Rename columns to match expected names
        df = df.rename(columns={v: k for k, v in column_mapping.items() if v in df.columns})
        
        # Convert date column if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        st.success(f"✅ Loaded {len(df)} records from Excel file")
        return df
        
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        # Fallback to sample data
        return load_sample_data_fallback()
