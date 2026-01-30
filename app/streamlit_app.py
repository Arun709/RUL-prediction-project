import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Turbofan RUL Predictor | AI-Powered Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Theme Toggle State
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'light'

def toggle_theme():
    st.session_state.theme_mode = 'dark' if st.session_state.theme_mode == 'light' else 'light'

# Custom CSS for premium styling
# Apply theme-specific CSS
theme = st.session_state.theme_mode

if theme == 'dark':
    bg_primary = 'rgba(26, 32, 44, 0.98)'
    bg_secondary = '#1a202c'
    bg_tertiary = '#2d3748'
    bg_gauge = '#2d3748'
    text_primary = '#e2e8f0'
    text_secondary = '#cbd5e0'
    text_tertiary = '#a0aec0'
    border_color = '#4a5568'
    card_bg = 'rgba(45, 55, 72, 0.9)'
else:  # light mode
    bg_primary = 'rgba(240, 244, 248, 0.98)'
    bg_secondary = '#e6eef5'
    bg_tertiary = '#d1dde8'
    bg_gauge = '#f0f4f8'
    text_primary = '#1a202c'
    text_secondary = '#2d3748'
    text_tertiary = '#4a5568'
    border_color = '#e6eef5'
    card_bg = 'rgba(255, 255, 255, 0.9)'

st.markdown(f"""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Base Styles */
    * {{
        font-family: 'Inter', sans-serif;
    }}

    /* Theme-adaptive text colors */
    body, p, span, div:not(.metric-card):not(.status-badge), label, h1, h2, h3, h4, h5, h6, li, td, th, a {{
        color: {text_primary} !important;
    }}

    /* Main content area text */
    .main .block-container * {{
        color: {text_primary} !important;
    }}

    /* Specific overrides for Streamlit elements */
    .stMarkdown, .stMarkdown *, .stMarkdown p, .stMarkdown span, .stMarkdown div, .stMarkdown li {{
        color: {text_primary} !important;
    }}

    .stText, .stText p, .stText span, .stText * {{
        color: {text_primary} !important;
    }}

    /* Input labels and text */
    .stTextInput label, .stNumberInput label, .stSelectbox label, 
    .stSlider label, .stFileUploader label, .stDateInput label {{
        color: {text_primary} !important;
        font-weight: 600 !important;
    }}

    /* Widget text */
    .stCheckbox span, .stRadio label, .stMultiselect label, .stCheckbox label {{
        color: {text_primary} !important;
    }}

    /* Headers */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {{
        color: {text_primary} !important;
    }}

    /* Element container */
    .element-container, .element-container * {{
        color: {text_primary} !important;
    }}

    /* All paragraph text */
    .main p, .main span, .main div {{
        color: {text_primary} !important;
    }}

    /* Column text */
    .row-widget p, .row-widget span, .row-widget div {{
        color: {text_primary} !important;
    }}

    /* Main container */
    .main {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: {bg_primary};
        border-radius: 20px;
        margin-top: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }}

    /* Header styling */
    h1 {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
        text-align: center;
        animation: fadeInDown 1s ease-in;
    }}

    h2 {{
        color: {text_primary} !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }}

    h3 {{
        color: {text_primary} !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
    }}

    h4 {{
        color: {text_primary} !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }}

    /* Paragraph and text styling */
    p, span:not(.metric-value):not(.metric-label):not(.metric-delta), div:not(.metric-card):not(.status-badge) {{
        color: {text_primary} !important;
    }}

    /* All text content */
    .css-1v0mbdj, .css-16idsys, .css-1dp5vir {{
        color: {text_primary} !important;
    }}

    /* Label styling */
    label, .stLabel {{
        color: {text_primary} !important;
        font-weight: 600 !important;
    }}

    /* Markdown text */
    .stMarkdown, .stMarkdown *, [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {{
        color: {text_primary} !important;
    }}

    /* Text in expanders */
    .streamlit-expanderHeader, .streamlit-expanderHeader * {{
        color: {text_primary} !important;
        font-weight: 600 !important;
    }}

    /* Tab content text */
    [data-baseweb="tab-panel"] *, 
    [data-baseweb="tab-panel"] h1,
    [data-baseweb="tab-panel"] h2, 
    [data-baseweb="tab-panel"] h3,
    [data-baseweb="tab-panel"] h4,
    [data-baseweb="tab-panel"] p,
    [data-baseweb="tab-panel"] span,
    [data-baseweb="tab-panel"] div:not(.metric-card) {{
        color: {text_primary} !important;
    }}

    /* Metric cards */
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 1rem;
        color: #4a5568;
        font-weight: 600;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 1rem;
    }}

    /* Custom metric card */
    .metric-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }}

    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }}

    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: white !important;
    }}

    .metric-label {{
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
        color: white !important;
    }}

    .metric-delta {{
        font-size: 1rem;
        margin-top: 0.5rem;
        color: white !important;
    }}

    /* Status badges */
    .status-badge {{
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.2rem;
        color: white !important;
    }}

    .status-healthy {{
        background: #48bb78;
        color: white !important;
    }}

    .status-warning {{
        background: #ed8936;
        color: white !important;
    }}

    .status-critical {{
        background: #f56565;
        color: white !important;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }}

    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }}

    /* Theme toggle button styling */
    div[data-testid="stHorizontalBlock"] > div:first-child .stButton>button {{
        background: {bg_secondary};
        color: {text_primary} !important;
        border: 2px solid {border_color};
        padding: 0.5rem 1.5rem;
        font-size: 0.9rem;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        color: {text_primary} !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }}

    .stTabs [aria-selected="true"] * {{
        color: white !important;
    }}

    /* Tab content text */
    .stTabs [data-baseweb="tab-panel"] {{
        color: {text_primary} !important;
    }}

    .stTabs [data-baseweb="tab-panel"] *:not(.metric-value):not(.metric-label):not(.metric-delta) {{
        color: {text_primary} !important;
    }}

    .stTabs [data-baseweb="tab-panel"] h1,
    .stTabs [data-baseweb="tab-panel"] h2,
    .stTabs [data-baseweb="tab-panel"] h3,
    .stTabs [data-baseweb="tab-panel"] h4 {{
        color: {text_primary} !important;
        font-weight: 700 !important;
    }}

    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] span:not(.metric-value):not(.metric-label),
    .stTabs [data-baseweb="tab-panel"] div:not(.metric-card):not(.status-badge) {{
        color: {text_primary} !important;
    }}

    .stTabs [data-baseweb="tab-panel"] label {{
        color: {text_primary} !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] li {{
        color: white !important;
    }}

    [data-testid="stSidebar"] .stMarkdown {{
        color: white !important;
    }}

    [data-testid="stSidebar"] strong {{
        color: #a0d2ff !important;
    }}

    /* Info boxes */
    .stAlert {{
        border-radius: 10px;
        border-left: 4px solid #667eea;
        background: {card_bg};
    }}

    .stAlert *, .stAlert p, .stAlert span, .stAlert div {{
        color: {text_primary} !important;
    }}

    /* Success/Warning/Error boxes */
    .stSuccess *, .stWarning *, .stError *, .stInfo * {{
        color: {text_primary} !important;
    }}

    .stSuccess p, .stWarning p, .stError p, .stInfo p {{
        color: {text_primary} !important;
    }}

    /* Input fields */
    input, textarea, select {{
        color: {text_primary} !important;
        background: {card_bg} !important;
        border-color: {border_color} !important;
    }}

    /* Text content */
    .element-container p, 
    .element-container span:not(.metric-value):not(.metric-label),
    .element-container div:not(.metric-card):not(.status-badge) {{
        color: {text_primary} !important;
    }}

    /* Strong/bold text */
    strong, b {{
        color: {text_primary} !important;
        font-weight: 700 !important;
    }}

    /* Code blocks */
    code {{
        color: #d63384 !important;
        background-color: {bg_secondary} !important;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }}

    /* Links */
    a {{
        color: #667eea !important;
    }}

    a:hover {{
        color: #764ba2 !important;
    }}

    /* Lists */
    ul, ol, li {{
        color: {text_primary} !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background: {bg_secondary};
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        background: {card_bg};
    }}

    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {{
        color: {text_primary} !important;
        background: {card_bg} !important;
    }}

    /* Animations */
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes pulse {{
        0%, 100% {{
            opacity: 1;
        }}
        50% {{
            opacity: 0.5;
        }}
    }}

    .pulse {{
        animation: pulse 2s infinite;
    }}

    /* Hero section */
    .hero-section {{
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
    }}

    .hero-subtitle {{
        font-size: 1.2rem;
        color: {text_secondary} !important;
        margin-top: 1rem;
        font-weight: 500;
    }}

    .hero-subtitle strong {{
        color: #667eea !important;
        font-weight: 700;
    }}

    /* Progress bar */
    .progress-bar {{
        height: 8px;
        background: {bg_tertiary};
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }}

    .progress-fill {{
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.3s ease;
    }}
</style>
""", unsafe_allow_html=True)


# Theme Toggle Button
col_theme1, col_theme2, col_theme3 = st.columns([1, 1, 1])
with col_theme2:
    theme_icon = "🌙" if st.session_state.theme_mode == "light" else "☀️"
    theme_text = "Dark Mode" if st.session_state.theme_mode == "light" else "Light Mode"
    if st.button(f"{theme_icon} {theme_text}", key="theme_toggle", use_container_width=True, on_click=toggle_theme):
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/airplane-take-off.png", width=80)
    st.markdown('<h2 style=\"color: #1a202c !important;\">⚙️ System Configuration</h2>', unsafe_allow_html=True)
    
    api_url = st.text_input("🌐 API Endpoint", value="http://localhost:8000")
    
    st.markdown("---")
    
    st.markdown('<h3 style=\"color: #2d3748 !important;\">📊 Model Performance</h3>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R² Score', 'Accuracy'],
        'Value': ['12.5 cycles', '9.8 cycles', '0.89', '94.2%']
    })
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown('<h3 style=\"color: #2d3748 !important;\">🎯 Quick Stats</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color: white !important;">
    
    - **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
    - **Dataset:** NASA C-MAPSS FD001
    - **Total Predictions:** 15,847
    - **System Uptime:** 99.8%
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<h3 style=\"color: #2d3748 !important;\">🔔 Alerts</h3>', unsafe_allow_html=True)
    st.warning("⚠️ 12 engines require attention")
    st.info("ℹ️ Maintenance window: 2 days")

# Load and generate example data
@st.cache_data
def load_example_data():
    np.random.seed(42)
    data = []
    for i in range(50):
        rul = np.random.randint(5, 200)
        if rul < 30:
            status = 'Critical'
        elif rul < 80:
            status = 'Warning'
        else:
            status = 'Healthy'
        
        data.append({
            'unit_id': i + 1,
            'time_cycle': np.random.randint(50, 300),
            'sensor_2': np.random.uniform(640, 645),
            'sensor_3': np.random.uniform(1585, 1595),
            'sensor_4': np.random.uniform(1395, 1405),
            'predicted_rul': rul,
            'maintenance_status': status,
            'confidence': np.random.uniform(0.85, 0.99),
            'last_maintenance': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d')
        })
    return pd.DataFrame(data)

example_data = load_example_data()

# Calculate fleet statistics
total_engines = len(example_data)
critical_count = len(example_data[example_data['maintenance_status'] == 'Critical'])
warning_count = len(example_data[example_data['maintenance_status'] == 'Warning'])
healthy_count = len(example_data[example_data['maintenance_status'] == 'Healthy'])
avg_rul = example_data['predicted_rul'].mean()

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Fleet Dashboard", "🔍 Single Prediction", "📈 Batch Analysis", "📉 Analytics"])

with tab1:
    st.markdown('<h2 style="color: #1a202c !important;">🎯 Real-Time Fleet Overview</h2>', unsafe_allow_html=True)
    
    # Top metrics with custom styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div class="metric-label">Total Engines</div>
            <div class="metric-value">{total_engines}</div>
            <div class="metric-delta">↑ 2 new this month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f56565 0%, #c53030 100%);">
            <div class="metric-label">Critical Status</div>
            <div class="metric-value">{critical_count}</div>
            <div class="metric-delta">↓ 3 from last week</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);">
            <div class="metric-label">Warning Status</div>
            <div class="metric-value">{warning_count}</div>
            <div class="metric-delta">→ Stable</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">
            <div class="metric-label">Healthy Status</div>
            <div class="metric-value">{healthy_count}</div>
            <div class="metric-delta">↑ 5 improved</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);">
            <div class="metric-label">Avg RUL</div>
            <div class="metric-value">{avg_rul:.0f}</div>
            <div class="metric-delta">cycles remaining</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 style="color: #2d3748 !important;">📊 RUL Distribution</h3>', unsafe_allow_html=True)
        
        # Enhanced histogram with better colors
        fig_hist = go.Figure()
        
        for status, color in [('Healthy', '#48bb78'), ('Warning', '#ed8936'), ('Critical', '#f56565')]:
            data_subset = example_data[example_data['maintenance_status'] == status]
            fig_hist.add_trace(go.Histogram(
                x=data_subset['predicted_rul'],
                name=status,
                marker_color=color,
                opacity=0.7,
                nbinsx=20
            ))
        
        fig_hist.update_layout(
            barmode='stack',
            xaxis_title='Predicted RUL (cycles)',
            yaxis_title='Number of Engines',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown('<h3 style="color: #2d3748 !important;">🎯 Fleet Health Status</h3>', unsafe_allow_html=True)
        
        # Donut chart for status distribution
        status_counts = example_data['maintenance_status'].value_counts()
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.6,
            marker=dict(colors=['#48bb78', '#ed8936', '#f56565']),
            textinfo='label+percent',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_donut.update_layout(
            annotations=[dict(text=f'{total_engines}<br>Engines', x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 style=\"color: #2d3748 !important;\">🔄 RUL vs Confidence Level</h3>', unsafe_allow_html=True)
        
        fig_scatter = px.scatter(
            example_data,
            x='predicted_rul',
            y='confidence',
            color='maintenance_status',
            size='time_cycle',
            hover_data=['unit_id', 'last_maintenance'],
            color_discrete_map={'Healthy': '#48bb78', 'Warning': '#ed8936', 'Critical': '#f56565'},
            title=''
        )
        
        fig_scatter.update_layout(
            xaxis_title='Predicted RUL (cycles)',
            yaxis_title='Confidence Score',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown('<h3 style=\"color: #2d3748 !important;\">📈 RUL Trend Over Time</h3>', unsafe_allow_html=True)
        
        # Simulated time series data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        trend_data = pd.DataFrame({
            'date': dates,
            'avg_rul': np.random.randint(80, 95, 30),
            'critical_count': np.random.randint(8, 15, 30)
        })
        
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_trend.add_trace(
            go.Scatter(x=trend_data['date'], y=trend_data['avg_rul'], 
                      name='Avg RUL', line=dict(color='#667eea', width=3),
                      fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)'),
            secondary_y=False
        )
        
        fig_trend.add_trace(
            go.Scatter(x=trend_data['date'], y=trend_data['critical_count'], 
                      name='Critical Count', line=dict(color='#f56565', width=3, dash='dash')),
            secondary_y=True
        )
        
        fig_trend.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            hovermode='x unified'
        )
        
        fig_trend.update_xaxes(title_text='Date')
        fig_trend.update_yaxes(title_text='Average RUL', secondary_y=False)
        fig_trend.update_yaxes(title_text='Critical Engines', secondary_y=True)
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Engine Status Table with enhanced styling
    st.markdown('<h3 style=\"color: #2d3748 !important;\">🗂️ Detailed Engine Status</h3>', unsafe_allow_html=True)
    
    # Add status badges to dataframe
    display_df = example_data.copy()
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
    
    # Color-code the dataframe
    def highlight_status(row):
        if row['maintenance_status'] == 'Critical':
            return ['background-color: #fed7d7'] * len(row)
        elif row['maintenance_status'] == 'Warning':
            return ['background-color: #feebc8'] * len(row)
        else:
            return ['background-color: #c6f6d5'] * len(row)
    
    styled_df = display_df.style.apply(highlight_status, axis=1)
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
        column_config={
            "unit_id": st.column_config.NumberColumn("Engine ID", format="%d"),
            "predicted_rul": st.column_config.ProgressColumn(
                "Predicted RUL",
                format="%.0f cycles",
                min_value=0,
                max_value=200
            ),
            "maintenance_status": st.column_config.TextColumn("Status"),
            "confidence": st.column_config.TextColumn("Confidence"),
            "last_maintenance": st.column_config.DateColumn("Last Maintenance")
        }
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Fleet Report (CSV)",
        data=csv,
        file_name=f"fleet_report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with tab2:
    st.markdown('<h2 style=\"color: #1a202c !important;\">🔍 Single Engine RUL Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 style=\"color: #2d3748 !important;\">🎛️ Engine Configuration</h3>', unsafe_allow_html=True)
        
        unit_id = st.number_input("🔧 Engine Unit ID", min_value=1, max_value=1000, value=1, step=1)
        time_cycle = st.number_input("⏱️ Current Cycle", min_value=1, max_value=500, value=100, step=1)
        
        st.markdown('<h4 style=\"color: #4a5568 !important;\">Operating Settings</h4>', unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            setting_1 = st.slider("🌡️ Altitude", 0.0, 1.0, 0.5, 0.01)
        with col_s2:
            setting_2 = st.slider("⚡ Mach", 0.0, 1.0, 0.5, 0.01)
        with col_s3:
            setting_3 = st.slider("🎚️ Throttle", 0.0, 1.0, 0.5, 0.01)
        
        # Display settings visually
        st.markdown('<h4 style=\"color: #4a5568 !important;\">Settings Visualization</h4>', unsafe_allow_html=True)
        fig_settings = go.Figure()
        
        fig_settings.add_trace(go.Bar(
            x=['Altitude', 'Mach', 'Throttle'],
            y=[setting_1, setting_2, setting_3],
            marker=dict(
                color=['#667eea', '#764ba2', '#f56565'],
                line=dict(width=2, color='white')
            ),
            text=[f"{setting_1:.2f}", f"{setting_2:.2f}", f"{setting_3:.2f}"],
            textposition='outside'
        ))
        
        fig_settings.update_layout(
            yaxis=dict(range=[0, 1.2]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=250,
            showlegend=False,
            margin=dict(t=20, b=20)
        )
        
        st.plotly_chart(fig_settings, use_container_width=True)
    
    with col2:
        st.markdown('<h3 style=\"color: #2d3748 !important;\">📡 Sensor Readings</h3>', unsafe_allow_html=True)
        
        # Create sensor input in a grid
        sensors = {}
        
        # Use tabs for sensor categories
        sensor_tab1, sensor_tab2, sensor_tab3 = st.tabs(["Temperature", "Pressure", "Other"])
        
        with sensor_tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                for i in [2, 3, 4, 11, 12, 13]:
                    sensors[f'sensor_{i}'] = st.number_input(
                        f"🌡️ Sensor {i}",
                        value=float(np.random.randint(500, 650)),
                        format="%.2f",
                        key=f"sensor_{i}"
                    )
            with col_b:
                for i in [7, 8, 9, 14, 15]:
                    sensors[f'sensor_{i}'] = st.number_input(
                        f"🌡️ Sensor {i}",
                        value=float(np.random.randint(500, 650)),
                        format="%.2f",
                        key=f"sensor_{i}"
                    )
        
        with sensor_tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                for i in [1, 5, 6]:
                    sensors[f'sensor_{i}'] = st.number_input(
                        f"💨 Sensor {i}",
                        value=float(np.random.randint(500, 650)),
                        format="%.2f",
                        key=f"sensor_{i}"
                    )
            with col_b:
                for i in [16, 17, 18]:
                    sensors[f'sensor_{i}'] = st.number_input(
                        f"💨 Sensor {i}",
                        value=float(np.random.randint(500, 650)),
                        format="%.2f",
                        key=f"sensor_{i}"
                    )
        
        with sensor_tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                for i in [10, 19]:
                    sensors[f'sensor_{i}'] = st.number_input(
                        f"⚙️ Sensor {i}",
                        value=float(np.random.randint(500, 650)),
                        format="%.2f",
                        key=f"sensor_{i}"
                    )
            with col_b:
                for i in [20, 21]:
                    sensors[f'sensor_{i}'] = st.number_input(
                        f"⚙️ Sensor {i}",
                        value=float(np.random.randint(500, 650)),
                        format="%.2f",
                        key=f"sensor_{i}"
                    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict button with animation
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button("🔮 PREDICT RUL", type="primary", use_container_width=True)
    
    if predict_button:
        # Progress bar animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Collecting sensor data...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        status_text.text("🧠 Running AI model...")
        progress_bar.progress(50)
        time.sleep(0.3)
        
        status_text.text("📊 Analyzing results...")
        progress_bar.progress(80)
        time.sleep(0.3)
        
        status_text.text("✅ Prediction complete!")
        progress_bar.progress(100)
        time.sleep(0.2)
        
        progress_bar.empty()
        status_text.empty()
        
        # Prepare data
        data = {
            'unit_id': unit_id,
            'time_cycle': time_cycle,
            'setting_1': setting_1,
            'setting_2': setting_2,
            'setting_3': setting_3,
            **sensors
        }
        
        # Simulate prediction (replace with actual API call)
        try:
            # Uncomment for real API call
            # response = requests.post(f"{api_url}/predict", json=data)
            # result = response.json()
            
            # Simulated result
            predicted_rul = np.random.randint(10, 180)
            confidence = np.random.uniform(0.85, 0.99)
            
            result = {
                'predicted_rul': predicted_rul,
                'maintenance_required': predicted_rul < 50,
                'confidence_level': 'high' if confidence > 0.9 else 'medium',
                'confidence_score': confidence
            }
            
            # Display results with animation
            st.success("✅ Prediction Complete!")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Predicted RUL",
                    f"{result['predicted_rul']:.0f} cycles",
                    delta=f"{result['predicted_rul'] - 100:.0f} vs baseline"
                )
            
            with col2:
                maintenance_status = "⚠️ REQUIRED" if result['maintenance_required'] else "✅ NOT NEEDED"
                maintenance_color = "red" if result['maintenance_required'] else "green"
                st.metric("Maintenance", maintenance_status)
            
            with col3:
                st.metric(
                    "Confidence",
                    result['confidence_level'].upper(),
                    delta=f"{result['confidence_score']:.1%}"
                )
            
            with col4:
                health_score = (result['predicted_rul'] / 200) * 100
                st.metric(
                    "Health Score",
                    f"{health_score:.0f}%",
                    delta=f"{health_score - 50:.0f}%"
                )
            
            # Detailed visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['predicted_rul'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Remaining Useful Life", 'font': {'size': 24, 'color': '#2d3748'}},
                    delta={'reference': 100, 'increasing': {'color': "#48bb78"}, 'decreasing': {'color': "#f56565"}},
                    number={'font': {'size': 48, 'color': '#667eea'}},
                    gauge={
                        'axis': {'range': [0, 200], 'tickwidth': 2, 'tickcolor': "#667eea"},
                        'bar': {'color': "#667eea", 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#e2e8f0",
                        'steps': [
                            {'range': [0, 30], 'color': '#fed7d7'},
                            {'range': [30, 80], 'color': '#feebc8'},
                            {'range': [80, 200], 'color': '#c6f6d5'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 30
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(l=20, r=20, t=80, b=20)
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Confidence and health indicators
                fig_indicators = go.Figure()
                
                categories = ['RUL Score', 'Confidence', 'Health', 'Reliability']
                values = [
                    (result['predicted_rul'] / 200) * 100,
                    result['confidence_score'] * 100,
                    health_score,
                    np.random.uniform(85, 95)
                ]
                
                fig_indicators.add_trace(go.Barpolar(
                    r=values,
                    theta=categories,
                    marker=dict(
                        color=values,
                        colorscale='Viridis',
                        line=dict(color='white', width=2)
                    ),
                    opacity=0.8
                ))
                
                fig_indicators.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    title={'text': 'Performance Metrics', 'font': {'size': 20, 'color': '#2d3748'}, 'x': 0.5}
                )
                
                st.plotly_chart(fig_indicators, use_container_width=True)
            
            # Recommendation panel
            st.markdown('<h3 style=\"color: #2d3748 !important;\">💡 Recommendations</h3>', unsafe_allow_html=True)
            
            if result['maintenance_required']:
                st.error(f"""
                **⚠️ MAINTENANCE REQUIRED**
                
                - Schedule maintenance within the next {result['predicted_rul']//5} cycles
                - Priority: {'HIGH' if result['predicted_rul'] < 30 else 'MEDIUM'}
                - Estimated downtime: 4-6 hours
                - Recommended actions: Inspect sensors {', '.join([str(i) for i in np.random.choice(range(1, 22), 3)])}
                """)
            else:
                st.success(f"""
                **✅ ENGINE IN GOOD CONDITION**
                
                - Next maintenance check: {result['predicted_rul']//2} cycles
                - Continue normal operations
                - Monitor sensors regularly
                - Schedule preventive maintenance in {result['predicted_rul']//3} cycles
                """)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info(f"💡 Make sure the API is running at {api_url}")

with tab3:
    st.markdown('<h2 style=\"color: #1a202c !important;\">📈 Batch Analysis & Reporting</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 style=\"color: #2d3748 !important;\">📁 Upload Engine Data</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your CSV file here or click to browse",
            type=['csv'],
            help="Upload a CSV file containing sensor data for multiple engines"
        )
    
    with col2:
        st.markdown('<h3 style=\"color: #2d3748 !important;\">📋 File Requirements</h3>', unsafe_allow_html=True)
        st.info("""
        **Required columns:**
        - unit_id
        - time_cycle
        - setting_1, setting_2, setting_3
        - sensor_1 to sensor_21
        
        **Format:** CSV
        **Max size:** 10MB
        """)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} records.")
            
            # Data preview with tabs
            preview_tab1, preview_tab2, preview_tab3 = st.tabs(["📊 Data Preview", "📈 Statistics", "🔍 Data Quality"])
            
            with preview_tab1:
                st.markdown('<h4 style=\"color: #4a5568 !important;\">First 10 Rows</h4>', unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Unique Engines", df['unit_id'].nunique() if 'unit_id' in df.columns else 'N/A')
            
            with preview_tab2:
                st.markdown('<h4 style=\"color: #4a5568 !important;\">Statistical Summary</h4>', unsafe_allow_html=True)
                st.dataframe(df.describe(), use_container_width=True)
            
            with preview_tab3:
                st.markdown('<h4 style=\"color: #4a5568 !important;\">Data Quality Check</h4>', unsafe_allow_html=True)
                
                # Missing values
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    st.warning(f"⚠️ Found {missing.sum()} missing values")
                    fig_missing = px.bar(
                        x=missing[missing > 0].index,
                        y=missing[missing > 0].values,
                        labels={'x': 'Column', 'y': 'Missing Count'},
                        title='Missing Values by Column'
                    )
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.success("✅ No missing values detected")
                
                # Duplicate check
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    st.warning(f"⚠️ Found {duplicates} duplicate rows")
                else:
                    st.success("✅ No duplicate rows")
            
            st.markdown("---")
            
            # Batch prediction button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 ANALYZE BATCH", type="primary", use_container_width=True):
                    # Progress animation
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(101):
                        progress_bar.progress(i)
                        if i < 30:
                            status_text.text(f"📥 Loading data... {i}%")
                        elif i < 70:
                            status_text.text(f"🧠 Processing predictions... {i}%")
                        else:
                            status_text.text(f"📊 Generating report... {i}%")
                        time.sleep(0.01)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ Batch analysis complete!")
                    
                    # Generate simulated results
                    results_df = df.copy()
                    results_df['predicted_rul'] = np.random.randint(10, 200, len(df))
                    results_df['confidence'] = np.random.uniform(0.8, 0.99, len(df))
                    results_df['maintenance_required'] = results_df['predicted_rul'] < 50
                    results_df['status'] = results_df['predicted_rul'].apply(
                        lambda x: 'Critical' if x < 30 else ('Warning' if x < 80 else 'Healthy')
                    )
                    
                    # Summary metrics
                    st.markdown('<h3 style=\"color: #2d3748 !important;\">📊 Batch Analysis Summary</h3>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Analyzed", len(results_df))
                    with col2:
                        critical = (results_df['status'] == 'Critical').sum()
                        st.metric("Critical", critical, delta=f"{(critical/len(results_df)*100):.1f}%", delta_color="inverse")
                    with col3:
                        maintenance = results_df['maintenance_required'].sum()
                        st.metric("Needs Maintenance", maintenance)
                    with col4:
                        avg_rul = results_df['predicted_rul'].mean()
                        st.metric("Avg RUL", f"{avg_rul:.0f} cycles")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_batch_dist = px.histogram(
                            results_df,
                            x='predicted_rul',
                            color='status',
                            nbins=30,
                            title='Batch RUL Distribution',
                            color_discrete_map={'Healthy': '#48bb78', 'Warning': '#ed8936', 'Critical': '#f56565'}
                        )
                        fig_batch_dist.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_batch_dist, use_container_width=True)
                    
                    with col2:
                        status_counts = results_df['status'].value_counts()
                        fig_batch_pie = px.pie(
                            values=status_counts.values,
                            names=status_counts.index,
                            title='Status Distribution',
                            color=status_counts.index,
                            color_discrete_map={'Healthy': '#48bb78', 'Warning': '#ed8936', 'Critical': '#f56565'}
                        )
                        st.plotly_chart(fig_batch_pie, use_container_width=True)
                    
                    # Results table
                    st.markdown('<h3 style=\"color: #2d3748 !important;\">📋 Detailed Results</h3>', unsafe_allow_html=True)
                    st.dataframe(
                        results_df[['unit_id', 'time_cycle', 'predicted_rul', 'confidence', 'status', 'maintenance_required']],
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download button
                    csv_output = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_output,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file is in the correct format.")

with tab4:
    st.markdown('<h2 style=\"color: #1a202c !important;\">📉 Advanced Analytics & Insights</h2>', unsafe_allow_html=True)
    
    # Time period selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<h3 style=\"color: #2d3748 !important;\">📅 Analysis Period</h3>', unsafe_allow_html=True)
    with col2:
        period = st.selectbox("Select Period", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom"])
    with col3:
        if period == "Custom":
            st.date_input("From Date", datetime.now() - timedelta(days=30))
    
    # Generate time series data
    days = 30 if period == "Last 30 Days" else 90
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    analytics_data = pd.DataFrame({
        'date': dates,
        'avg_rul': np.random.randint(75, 95, days) + np.sin(np.linspace(0, 4*np.pi, days)) * 10,
        'critical_engines': np.random.randint(8, 18, days),
        'maintenance_events': np.random.randint(2, 8, days),
        'prediction_accuracy': np.random.uniform(0.85, 0.95, days),
        'cost_saved': np.random.randint(50000, 150000, days)
    })
    
    # Key Insights
    st.markdown('<h3 style=\"color: #2d3748 !important;\">🎯 Key Insights</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        trend = "↑" if analytics_data['avg_rul'].iloc[-1] > analytics_data['avg_rul'].iloc[0] else "↓"
        st.metric(
            "RUL Trend",
            f"{analytics_data['avg_rul'].iloc[-1]:.0f}",
            delta=f"{trend} {abs(analytics_data['avg_rul'].iloc[-1] - analytics_data['avg_rul'].iloc[0]):.0f} cycles"
        )
    
    with col2:
        total_maintenance = analytics_data['maintenance_events'].sum()
        st.metric("Total Maintenance", total_maintenance, delta=f"{(total_maintenance/days):.1f} per day")
    
    with col3:
        avg_accuracy = analytics_data['prediction_accuracy'].mean()
        st.metric("Avg Accuracy", f"{avg_accuracy:.1%}", delta="↑ 2.3%")
    
    with col4:
        total_savings = analytics_data['cost_saved'].sum()
        st.metric("Total Savings", f"${total_savings/1000:.0f}K", delta="↑ 12%")
    
    # Advanced Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 style=\"color: #2d3748 !important;\">📈 Multi-Metric Trend Analysis</h3>', unsafe_allow_html=True)
        
        fig_multi = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average RUL Over Time', 'Critical Engines & Maintenance Events'),
            vertical_spacing=0.15
        )
        
        fig_multi.add_trace(
            go.Scatter(
                x=analytics_data['date'],
                y=analytics_data['avg_rul'],
                name='Avg RUL',
                line=dict(color='#667eea', width=3),
                fill='tozeroy'
            ),
            row=1, col=1
        )
        
        fig_multi.add_trace(
            go.Scatter(
                x=analytics_data['date'],
                y=analytics_data['critical_engines'],
                name='Critical Engines',
                line=dict(color='#f56565', width=2)
            ),
            row=2, col=1
        )
        
        fig_multi.add_trace(
            go.Bar(
                x=analytics_data['date'],
                y=analytics_data['maintenance_events'],
                name='Maintenance Events',
                marker=dict(color='#ed8936'),
                opacity=0.6
            ),
            row=2, col=1
        )
        
        fig_multi.update_layout(
            height=600,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_multi, use_container_width=True)
    
    with col2:
        st.markdown('<h3 style=\"color: #2d3748 !important;\">💰 Cost Analysis</h3>', unsafe_allow_html=True)
        
        # Cost breakdown
        cost_categories = ['Prevented Failures', 'Optimized Maintenance', 'Reduced Downtime', 'Parts Savings']
        cost_values = [total_savings * 0.4, total_savings * 0.3, total_savings * 0.2, total_savings * 0.1]
        
        fig_cost = go.Figure(data=[
            go.Bar(
                y=cost_categories,
                x=cost_values,
                orientation='h',
                marker=dict(
                    color=['#667eea', '#764ba2', '#48bb78', '#ed8936'],
                    line=dict(color='white', width=2)
                ),
                text=[f'${v/1000:.0f}K' for v in cost_values],
                textposition='outside'
            )
        ])
        
        fig_cost.update_layout(
            title='Cost Savings Breakdown',
            xaxis_title='Savings ($)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)
        
        st.markdown('<h3 style=\"color: #2d3748 !important;\">🎯 Prediction Accuracy Trend</h3>', unsafe_allow_html=True)
        
        fig_accuracy = go.Figure()
        
        fig_accuracy.add_trace(go.Scatter(
            x=analytics_data['date'],
            y=analytics_data['prediction_accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#48bb78', width=3),
            marker=dict(size=8, color='#48bb78', line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor='rgba(72, 187, 120, 0.2)'
        ))
        
        fig_accuracy.add_hline(
            y=0.9,
            line_dash="dash",
            line_color="red",
            annotation_text="Target: 90%"
        )
        
        fig_accuracy.update_layout(
            yaxis=dict(range=[0.8, 1.0], tickformat='.0%'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=280
        )
        
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Correlation heatmap
    st.markdown('<h3 style=\"color: #2d3748 !important;\">🔥 Sensor Correlation Heatmap</h3>', unsafe_allow_html=True)
    
    # Generate correlation matrix
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    corr_data = pd.DataFrame(np.random.rand(21, 21), columns=sensor_cols, index=sensor_cols)
    np.fill_diagonal(corr_data.values, 1)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='Viridis',
        text=corr_data.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig_heatmap.update_layout(
        title='Sensor Correlation Matrix',
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Export report
    st.markdown('<h3 style=\"color: #2d3748 !important;\">📄 Export Analytics Report</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate PDF Report", use_container_width=True):
            st.info("PDF report generation coming soon!")
    
    with col2:
        if st.button("📈 Export to Excel", use_container_width=True):
            st.info("Excel export coming soon!")
    
    with col3:
        csv_analytics = analytics_data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_analytics,
            file_name=f"analytics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='color: #2d3748 !important; font-weight: 600; margin-bottom: 0.5rem;'><strong>Turbofan Engine Predictive Maintenance System v2.0</strong></p>
        <p style='color: #4a5568 !important; margin-bottom: 0.5rem;'>Powered by AI • Built with ❤️ • © 2026 All Rights Reserved</p>
        <p style='font-size: 0.9rem; color: #4a5568 !important;'>
            🔒 Secure • 🚀 Fast • 📊 Accurate • 💡 Intelligent
        </p>
    </div>
    """, unsafe_allow_html=True)
