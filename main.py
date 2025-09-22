"""
ARGO Float Data Discovery Dashboard
A production-ready Streamlit application for exploring oceanographic data
with AI-powered conversational interface and geospatial visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import xarray as xr
from datetime import datetime, timedelta
import tempfile
import os
import sqlite3
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ARGO Float Data Discovery",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ocean-inspired theme
st.markdown("""
<style>
    .main > div {
        padding: 1rem 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .chat-message {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1e90ff;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class ARGODataProcessor:
    """Handles ARGO float data processing and database operations."""
    
    def __init__(self):
        self.db_path = ":memory:"  # In-memory database for demo
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for storing ARGO data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for ARGO data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS argo_profiles (
                id INTEGER PRIMARY KEY,
                float_id TEXT,
                cycle_number INTEGER,
                date TEXT,
                latitude REAL,
                longitude REAL,
                temperature TEXT,  -- JSON array
                salinity TEXT,     -- JSON array
                pressure TEXT,     -- JSON array
                oxygen TEXT        -- JSON array
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def generate_dummy_data(self, n_floats=50, n_profiles_per_float=10) -> pd.DataFrame:
        """Generate realistic dummy ARGO float data."""
        np.random.seed(42)
        
        data = []
        regions = {
            'North Atlantic': {'lat': (40, 60), 'lon': (-50, -10)},
            'Arabian Sea': {'lat': (10, 25), 'lon': (60, 75)},
            'Pacific Equatorial': {'lat': (-10, 10), 'lon': (140, 180)},
            'Southern Ocean': {'lat': (-60, -40), 'lon': (0, 360)}
        }
        
        for float_id in range(1, n_floats + 1):
            # Assign float to a region
            region = np.random.choice(list(regions.keys()))
            lat_range = regions[region]['lat']
            lon_range = regions[region]['lon']
            
            base_lat = np.random.uniform(lat_range[0], lat_range[1])
            base_lon = np.random.uniform(lon_range[0], lon_range[1])
            
            for cycle in range(1, n_profiles_per_float + 1):
                # Add some drift to position
                lat = base_lat + np.random.normal(0, 0.5)
                lon = base_lon + np.random.normal(0, 0.5)
                
                # Generate date (last 2 years)
                days_ago = np.random.randint(0, 730)
                date = datetime.now() - timedelta(days=days_ago)
                
                # Generate depth profiles (0-2000m)
                depths = np.arange(0, 2000, 10)
                n_depths = len(depths)
                
                # Temperature profile (decreases with depth)
                surface_temp = 15 + 10 * np.exp(-abs(lat)/30)  # Warmer near equator
                temp_profile = surface_temp * np.exp(-depths/1000) + np.random.normal(0, 0.5, n_depths)
                
                # Salinity profile (varies with depth and region)
                base_salinity = 34.5 + np.random.normal(0, 0.5)
                sal_profile = base_salinity + 0.5 * np.sin(depths/500) + np.random.normal(0, 0.1, n_depths)
                
                # Oxygen profile (decreases with depth)
                surface_oxygen = 250 + np.random.normal(0, 20)
                oxy_profile = surface_oxygen * np.exp(-depths/800) + np.random.normal(0, 5, n_depths)
                
                data.append({
                    'float_id': f'ARGO_{float_id:04d}',
                    'cycle_number': cycle,
                    'date': date.strftime('%Y-%m-%d'),
                    'latitude': lat,
                    'longitude': lon,
                    'temperature': json.dumps(temp_profile.tolist()),
                    'salinity': json.dumps(sal_profile.tolist()),
                    'pressure': json.dumps(depths.tolist()),
                    'oxygen': json.dumps(oxy_profile.tolist()),
                    'region': region
                })
        
        return pd.DataFrame(data)
    
    def load_data_to_db(self, df: pd.DataFrame):
        """Load DataFrame to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        df_db = df.drop('region', axis=1, errors='ignore')  # Remove region column for DB
        df_db.to_sql('argo_profiles', conn, if_exists='replace', index=False)
        conn.close()
    
    def query_database(self, query: str) -> pd.DataFrame:
        """Execute SQL query on the database."""
        conn = sqlite3.connect(self.db_path)
        try:
            result = pd.read_sql_query(query, conn)
        except Exception as e:
            st.error(f"Database query error: {e}")
            result = pd.DataFrame()
        finally:
            conn.close()
        return result

class ConversationalAI:
    """Simple AI query processor for natural language to SQL translation."""
    
    def __init__(self):
        self.query_patterns = {
            'temperature': ['temperature', 'temp', 'thermal'],
            'salinity': ['salinity', 'salt', 'saline'],
            'oxygen': ['oxygen', 'o2', 'dissolved oxygen'],
            'location': ['near', 'around', 'in', 'region', 'area'],
            'time': ['last', 'recent', 'since', 'during', 'month', 'year'],
            'comparison': ['compare', 'difference', 'vs', 'versus'],
            'average': ['average', 'mean', 'typical'],
            'maximum': ['max', 'maximum', 'highest'],
            'minimum': ['min', 'minimum', 'lowest']
        }
    
    def parse_natural_query(self, user_query: str) -> Dict:
        """Parse natural language query into structured parameters."""
        query_lower = user_query.lower()
        
        parsed = {
            'parameters': [],
            'location': None,
            'time_filter': None,
            'operation': 'select',
            'sql_query': ''
        }
        
        # Detect parameters
        for param, keywords in self.query_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                if param in ['temperature', 'salinity', 'oxygen']:
                    parsed['parameters'].append(param)
        
        # Generate basic SQL query
        if 'salinity' in query_lower and 'equator' in query_lower:
            parsed['sql_query'] = """
                SELECT float_id, latitude, longitude, date, salinity 
                FROM argo_profiles 
                WHERE latitude BETWEEN -5 AND 5 
                ORDER BY date DESC 
                LIMIT 20
            """
        elif 'temperature' in query_lower and 'arabian' in query_lower:
            parsed['sql_query'] = """
                SELECT float_id, latitude, longitude, date, temperature 
                FROM argo_profiles 
                WHERE latitude BETWEEN 10 AND 25 AND longitude BETWEEN 60 AND 75
                ORDER BY date DESC 
                LIMIT 20
            """
        else:
            # Default query
            parsed['sql_query'] = """
                SELECT float_id, latitude, longitude, date 
                FROM argo_profiles 
                ORDER BY date DESC 
                LIMIT 10
            """
        
        return parsed

def create_world_map(df: pd.DataFrame, parameter: str = 'temperature') -> folium.Map:
    """Create interactive world map with ARGO float positions."""
    
    # Create base map
    m = folium.Map(
        location=[20, 0], 
        zoom_start=2,
        tiles='OpenStreetMap',
        width='100%',
        height='600px'
    )
    
    # Add custom tile layers
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap',
        name='Topographic',
        overlay=False,
    ).add_to(m)
    
    # Color map for different parameters
    color_maps = {
        'temperature': 'YlOrRd',
        'salinity': 'viridis',
        'oxygen': 'plasma'
    }
    
    # Get latest position for each float
    latest_positions = df.groupby('float_id').apply(
        lambda x: x.loc[x['date'].idxmax()]
    ).reset_index(drop=True)
    
    # Add markers
    for _, row in latest_positions.iterrows():
        try:
            # Get parameter value (average of profile)
            if parameter in row and row[parameter]:
                param_data = json.loads(row[parameter])
                param_value = np.mean(param_data) if param_data else 0
            else:
                param_value = 0
            
            # Determine color based on parameter value
            if parameter == 'temperature':
                color = 'red' if param_value > 20 else 'orange' if param_value > 10 else 'blue'
            elif parameter == 'salinity':
                color = 'purple' if param_value > 35 else 'blue' if param_value > 34 else 'green'
            else:
                color = 'darkblue'
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                popup=f"""
                <b>Float:</b> {row['float_id']}<br>
                <b>Date:</b> {row['date']}<br>
                <b>Lat:</b> {row['latitude']:.2f}<br>
                <b>Lon:</b> {row['longitude']:.2f}<br>
                <b>Avg {parameter.title()}:</b> {param_value:.2f}
                """,
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
            
        except Exception as e:
            continue
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_profile_plot(df: pd.DataFrame, float_ids: List[str], parameter: str = 'temperature') -> go.Figure:
    """Create depth profile plots for selected floats."""
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, float_id in enumerate(float_ids[:10]):  # Limit to 10 floats
        float_data = df[df['float_id'] == float_id].iloc[0]
        
        try:
            if parameter in float_data and float_data[parameter]:
                param_values = json.loads(float_data[parameter])
                pressures = json.loads(float_data['pressure'])
                
                fig.add_trace(go.Scatter(
                    x=param_values,
                    y=pressures,
                    mode='lines+markers',
                    name=f'{float_id}',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{float_id}</b><br>' +
                                f'{parameter.title()}: %{{x:.2f}}<br>' +
                                'Pressure: %{y:.0f} dbar<extra></extra>'
                ))
        except Exception as e:
            continue
    
    # Invert y-axis (depth increases downward)
    fig.update_layout(
        title=f'{parameter.title()} Profiles',
        xaxis_title=f'{parameter.title()} ({get_parameter_units(parameter)})',
        yaxis_title='Pressure (dbar)',
        yaxis=dict(autorange='reversed'),
        height=600,
        showlegend=True,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig

def create_time_series_plot(df: pd.DataFrame, float_id: str, parameter: str = 'temperature') -> go.Figure:
    """Create time series plot for a specific float."""
    
    float_data = df[df['float_id'] == float_id].sort_values('date')
    
    fig = go.Figure()
    
    dates = []
    surface_values = []
    deep_values = []
    
    for _, row in float_data.iterrows():
        try:
            if parameter in row and row[parameter]:
                param_values = json.loads(row[parameter])
                pressures = json.loads(row['pressure'])
                
                # Surface value (top 50m)
                surface_idx = np.where(np.array(pressures) <= 50)[0]
                surface_val = np.mean([param_values[i] for i in surface_idx]) if len(surface_idx) > 0 else np.nan
                
                # Deep value (>500m)
                deep_idx = np.where(np.array(pressures) >= 500)[0]
                deep_val = np.mean([param_values[i] for i in deep_idx]) if len(deep_idx) > 0 else np.nan
                
                dates.append(row['date'])
                surface_values.append(surface_val)
                deep_values.append(deep_val)
        except:
            continue
    
    # Surface time series
    fig.add_trace(go.Scatter(
        x=dates,
        y=surface_values,
        mode='lines+markers',
        name=f'Surface (<50m)',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=6)
    ))
    
    # Deep time series
    fig.add_trace(go.Scatter(
        x=dates,
        y=deep_values,
        mode='lines+markers',
        name=f'Deep (>500m)',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'{parameter.title()} Time Series - {float_id}',
        xaxis_title='Date',
        yaxis_title=f'{parameter.title()} ({get_parameter_units(parameter)})',
        height=400,
        template='plotly_white',
        hovermode='x'
    )
    
    return fig

def get_parameter_units(parameter: str) -> str:
    """Get units for different parameters."""
    units = {
        'temperature': '°C',
        'salinity': 'PSU',
        'oxygen': 'μmol/kg',
        'pressure': 'dbar'
    }
    return units.get(parameter, '')

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = ARGODataProcessor()
        st.session_state.ai_assistant = ConversationalAI()
        st.session_state.chat_history = []
    
    # Header
    st.title("🌊 ARGO Float Data Discovery Dashboard")
    st.markdown("*AI-powered oceanographic data exploration with interactive visualizations*")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload NetCDF File",
        type=['nc', 'netcdf'],
        help="Upload ARGO float NetCDF data file"
    )
    
    # Load dummy data if no file uploaded
    if 'df' not in st.session_state:
        with st.spinner("Loading sample ARGO data..."):
            st.session_state.df = st.session_state.data_processor.generate_dummy_data()
            st.session_state.data_processor.load_data_to_db(st.session_state.df)
        st.sidebar.success("✅ Sample data loaded!")
    
    # Parameter selection
    parameter = st.sidebar.selectbox(
        "Select Parameter",
        ['temperature', 'salinity', 'oxygen'],
        help="Choose oceanographic parameter to visualize"
    )
    
    # Region selector
    regions = ['All Regions', 'North Atlantic', 'Arabian Sea', 'Pacific Equatorial', 'Southern Ocean']
    selected_region = st.sidebar.selectbox("Select Region", regions)
    
    # Time range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Filter data
    df = st.session_state.df.copy()
    if selected_region != 'All Regions':
        df = df[df['region'] == selected_region]
    
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    
    # Main dashboard tabs
    tab1, tab2, tab3 = st.tabs(["🌍 Map View", "📊 Profile Plots", "💬 Chat Interface"])
    
    with tab1:
        st.header("Interactive ARGO Float Map")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Floats", len(df['float_id'].unique()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Profiles", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            latest_date = df['date'].max().strftime('%Y-%m-%d')
            st.metric("Latest Data", latest_date)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            date_range = (df['date'].max() - df['date'].min()).days
            st.metric("Date Range (days)", date_range)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create and display map
        if not df.empty:
            world_map = create_world_map(df, parameter)
            map_data = st_folium(world_map, width=None, height=600)
        else:
            st.warning("No data available for selected filters.")
    
    with tab2:
        st.header("Oceanographic Profile Analysis")
        
        # Float selection
        available_floats = df['float_id'].unique()
        selected_floats = st.multiselect(
            "Select Floats for Profile Comparison",
            available_floats,
            default=available_floats[:5] if len(available_floats) >= 5 else available_floats
        )
        
        if selected_floats:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Profile plot
                profile_fig = create_profile_plot(df, selected_floats, parameter)
                st.plotly_chart(profile_fig, use_container_width=True)
            
            with col2:
                # Time series for selected float
                if len(selected_floats) > 0:
                    selected_float = st.selectbox("Select Float for Time Series", selected_floats)
                    time_series_fig = create_time_series_plot(df, selected_float, parameter)
                    st.plotly_chart(time_series_fig, use_container_width=True)
            
            # Data table
            st.subheader("Selected Float Data")
            display_df = df[df['float_id'].isin(selected_floats)][
                ['float_id', 'date', 'latitude', 'longitude', 'cycle_number']
            ].sort_values('date', ascending=False)
            st.dataframe(display_df, use_container_width=True)
    
    with tab3:
        st.header("AI-Powered Data Query Interface")
        st.markdown("Ask questions about the ARGO float data in natural language!")
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            - "Show me salinity profiles near the equator"
            - "Find temperature data in the Arabian Sea"
            - "What's the average oxygen level in recent profiles?"
            - "Compare salinity between different regions"
            """)
        
        # Chat interface
        if st.session_state.chat_history:
            for i, (user_msg, ai_response) in enumerate(st.session_state.chat_history):
                st.markdown(f'<div class="chat-message"><b>You:</b> {user_msg}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message"><b>AI:</b> {ai_response}</div>', unsafe_allow_html=True)
        
        # Chat input
        user_query = st.chat_input("Ask about ARGO float data...")
        
        if user_query:
            # Process query
            parsed_query = st.session_state.ai_assistant.parse_natural_query(user_query)
            
            try:
                # Execute SQL query
                result_df = st.session_state.data_processor.query_database(parsed_query['sql_query'])
                
                if not result_df.empty:
                    ai_response = f"Found {len(result_df)} matching profiles. Here's what I discovered:"
                    
                    # Display results
                    st.markdown(f'<div class="chat-message"><b>You:</b> {user_query}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-message"><b>AI:</b> {ai_response}</div>', unsafe_allow_html=True)
                    
                    # Show data table
                    st.dataframe(result_df.head(10), use_container_width=True)
                    
                    # Create visualization if applicable
                    if len(result_df) > 0 and any(param in parsed_query['parameters'] for param in ['temperature', 'salinity', 'oxygen']):
                        param = next((p for p in parsed_query['parameters'] if p in ['temperature', 'salinity', 'oxygen']), 'temperature')
                        
                        # Create scatter plot of locations
                        if 'latitude' in result_df.columns and 'longitude' in result_df.columns:
                            fig = px.scatter_geo(
                                result_df,
                                lat='latitude',
                                lon='longitude',
                                hover_data=['float_id', 'date'],
                                title=f"Locations of Matching {param.title()} Profiles"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    ai_response = "No data found matching your query. Try adjusting your search criteria."
                    st.markdown(f'<div class="chat-message"><b>You:</b> {user_query}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-message"><b>AI:</b> {ai_response}</div>', unsafe_allow_html=True)
                
                # Add to chat history
                st.session_state.chat_history.append((user_query, ai_response))
                
            except Exception as e:
                error_response = f"Sorry, I encountered an error processing your query: {str(e)}"
                st.markdown(f'<div class="chat-message"><b>You:</b> {user_query}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message"><b>AI:</b> {error_response}</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append((user_query, error_response))
    
    # Footer
    st.markdown("---")
    st.markdown(
        "*ARGO Float Data Discovery Dashboard - Powered by Streamlit & AI*  \n"
        "🌊 Exploring the oceans, one profile at a time"
    )

if __name__ == "__main__":
    main()
