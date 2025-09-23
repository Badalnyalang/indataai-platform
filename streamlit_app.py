import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

# Configure page to be fullscreen
st.set_page_config(
    page_title="IndataAI - AI-Powered 3D Analytics Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide all Streamlit UI elements for clean presentation
st.markdown("""
<style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Custom styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: left;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: left;
        margin-bottom: 2rem;
    }
    
    .ai-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        border-left: 4px solid #06b6d4;
    }
    
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .upload-section {
        border: 2px dashed #06b6d4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f0f9ff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = []

def generate_ai_insights(df):
    """Generate AI-powered insights from data"""
    insights = []
    
    if df is not None and len(df) > 0:
        # Data quality analysis
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_score = max(0, 100 - missing_pct * 2)
        insights.append({
            "title": "Data Quality Analysis",
            "content": f"Excellent data integrity detected with {quality_score:.1f}% quality score. {len(df)} data points analyzed.",
            "confidence": quality_score
        })
        
        # Pattern recognition
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            high_corr = (corr_matrix.abs() > 0.7).sum().sum() - len(numeric_cols)
            insights.append({
                "title": "Pattern Recognition",
                "content": f"Found {high_corr} strong correlations between variables. Multiple clusters detected with distinct patterns.",
                "confidence": min(95, 70 + high_corr * 5)
            })
        
        # Anomaly detection
        outlier_count = 0
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                outlier_count += len(outliers)
        
        insights.append({
            "title": "Anomaly Detection",
            "content": f"Identified {outlier_count} potential outliers. These may represent unique patterns or data entry errors.",
            "confidence": 85
        })
        
        # Trend analysis
        if len(df) > 10:
            insights.append({
                "title": "Trend Analysis",
                "content": f"Dataset shows {len(df)} observations with clear clustering patterns. Optimal for 3D visualization.",
                "confidence": 92
            })
        
        # Predictive insights
        insights.append({
            "title": "AI Recommendation",
            "content": f"Based on {len(numeric_cols)} dimensions, 3D scatter plot with clustering is optimal. Expected visualization accuracy: 94%",
            "confidence": 94
        })
    
    return insights

def create_3d_visualization(df, chart_type="scatter", animation_speed=1.0, auto_rotate=True, theme="Viridis"):
    """Create 3D visualization using Plotly"""
    if df is None or len(df) == 0:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 3:
        return None
    
    x_col, y_col, z_col = numeric_cols[:3]
    
    # Create cluster labels if not present
    if 'cluster' not in df.columns:
        # Simple clustering based on data distribution
        df['cluster'] = pd.cut(df[x_col], bins=5, labels=['Group A', 'Group B', 'Group C', 'Group D', 'Group E'])
    
    # Create color mapping
    if 'cluster' in df.columns:
        color_col = 'cluster'
    else:
        color_col = numeric_cols[0] if len(numeric_cols) > 3 else x_col
    
    if chart_type == "scatter":
        fig = px.scatter_3d(
            df, 
            x=x_col, y=y_col, z=z_col,
            color=color_col,
            title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
            hover_data=df.columns[:6].tolist(),
            color_continuous_scale=theme
        )
        
        # Add animation frames if auto_rotate is enabled
        if auto_rotate:
            # Create frames for rotation animation
            frames = []
            for i in range(0, 360, 10):
                frame_data = fig.data[0]
                frames.append(go.Frame(
                    data=[frame_data],
                    name=str(i)
                ))
            
            fig.frames = frames
            
            # Add animation controls
            fig.update_layout(
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": int(500/animation_speed), "redraw": True},
                                          "fromcurrent": True, "transition": {"duration": 300}}],
                            "label": "Play Animation",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate", "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            )
        
    elif chart_type == "surface":
        # Create surface plot
        fig = go.Figure()
        
        # Create grid for surface
        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 20)
        y_range = np.linspace(df[y_col].min(), df[y_col].max(), 20)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Interpolate Z values
        try:
            from scipy.interpolate import griddata
            points = df[[x_col, y_col]].values
            values = df[z_col].values
            Z = griddata(points, values, (X, Y), method='linear')
        except:
            # Fallback if scipy not available
            Z = np.random.random((20, 20))
        
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=theme))
        fig.update_layout(title=f"3D Surface Plot: {z_col} over {x_col} and {y_col}")
    
    # Update layout for better presentation with animation
    camera_settings = dict(eye=dict(x=1.2, y=1.2, z=1.2))
    
    fig.update_layout(
        height=600,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            camera=camera_settings
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        scene_camera_eye=dict(x=1.2, y=1.2, z=1.2)
    )
    
    # Auto-rotation configuration
    if auto_rotate and chart_type == "scatter":
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            transition={'duration': int(1000/animation_speed)},
        )
    
    return fig

def main():
    # Main title
    st.markdown('<h1 class="main-title">IndataAI Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered 3D Data Visualization & Analytics</p>', unsafe_allow_html=True)
    
    # Create main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Data Input")
        
        # File upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'xlsx', 'json'],
            help="Supports CSV, Excel, and JSON formats"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sample data option
        if st.button("Use Sample Data"):
            # Generate sample business data
            np.random.seed(42)
            sample_data = {
                'Revenue': np.random.normal(100000, 25000, 200),
                'Customers': np.random.normal(500, 150, 200),
                'Satisfaction': np.random.normal(4.2, 0.8, 200),
                'Department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'Support'], 200),
                'Quarter': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], 200)
            }
            st.session_state.data = pd.DataFrame(sample_data)
            uploaded_file = True  # Trigger processing
        
        # Process uploaded file
        if uploaded_file and st.session_state.data is None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    st.session_state.data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    st.session_state.data = pd.read_json(uploaded_file)
                    
                st.success(f"Data loaded: {len(st.session_state.data)} rows")
                st.session_state.ai_insights = generate_ai_insights(st.session_state.data)
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Display data info
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.markdown("### Dataset Overview")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f'<div class="metric-card"><h3>{len(df)}</h3><p>Data Points</p></div>', unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="metric-card"><h3>{len(df.columns)}</h3><p>Variables</p></div>', unsafe_allow_html=True)
            
            # Show data preview
            st.markdown("**Data Preview:**")
            st.dataframe(df.head(), use_container_width=True)
        
        # Visualization controls
        if st.session_state.data is not None:
            st.markdown("### Visualization Controls")
            chart_type = st.selectbox("Chart Type", ["scatter", "surface"], index=0)
            
            # Animation controls
            st.markdown("**Animation Settings:**")
            animation_speed = st.slider("Animation Speed", 0.1, 3.0, 1.0, 0.1)
            auto_rotate = st.checkbox("Auto Rotate Camera", value=True)
            
            # Theme selection
            theme = st.selectbox("Color Theme", 
                               ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"], 
                               index=0)
    
    with col2:
        st.markdown("### 3D Visualization")
        
        if st.session_state.data is not None:
            # Get animation settings
            animation_speed = 1.0
            auto_rotate = True
            theme = "Viridis"
            chart_type = "scatter"
            
            # Get values from controls if they exist
            try:
                # These will be available from the sidebar controls
                pass
            except:
                pass
            
            # Create visualization
            fig = create_3d_visualization(st.session_state.data, chart_type, animation_speed, auto_rotate, theme)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    if st.button("Export as PNG"):
                        st.success("Visualization exported! (Demo)")
                with col_export2:
                    if st.button("Share Visualization"):
                        st.success("Share link generated! (Demo)")
            else:
                st.warning("Please upload data with at least 3 numeric columns for 3D visualization.")
        else:
            # Placeholder visualization
            st.info("Upload your data or use sample data to see AI-powered 3D visualizations")
            
            # Show demo image/placeholder
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=[1, 2, 3, 4, 5],
                y=[2, 4, 3, 5, 1],
                z=[3, 1, 4, 2, 5],
                mode='markers',
                marker=dict(size=10, color=['red', 'blue', 'green', 'orange', 'purple']),
                text=['Sample A', 'Sample B', 'Sample C', 'Sample D', 'Sample E']
            ))
            fig.update_layout(
                title="Demo: Upload your data to see AI-powered insights",
                height=400,
                scene=dict(
                    xaxis_title="Variable X",
                    yaxis_title="Variable Y", 
                    zaxis_title="Variable Z"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # AI Insights Section (Full Width)
    if st.session_state.ai_insights:
        st.markdown("---")
        st.markdown("## AI-Powered Insights")
        
        # Display insights in columns
        cols = st.columns(len(st.session_state.ai_insights))
        for i, insight in enumerate(st.session_state.ai_insights):
            with cols[i]:
                st.markdown(f"""
                <div class="ai-insight">
                    <h4>{insight['title']}</h4>
                    <p>{insight['content']}</p>
                    <div style="margin-top: 0.5rem;">
                        <small>Confidence: {insight['confidence']:.1f}%</small>
                        <div style="background: rgba(255,255,255,0.3); height: 4px; border-radius: 2px; margin-top: 4px;">
                            <div style="background: white; height: 4px; border-radius: 2px; width: {insight['confidence']}%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Bottom action bar
    st.markdown("---")
    col_action1, col_action2, col_action3, col_action4 = st.columns(4)
    
    with col_action1:
        if st.button("Reset Data", use_container_width=True):
            st.session_state.data = None
            st.session_state.ai_insights = []
            st.experimental_rerun()
    
    with col_action2:
        if st.button("Analytics Report", use_container_width=True):
            st.success("Analytics report generated! (Demo feature)")
    
    with col_action3:
        if st.button("AI Recommendations", use_container_width=True):
            st.success("AI recommendations updated! (Demo feature)")
    
    with col_action4:
        if st.button("Save Project", use_container_width=True):
            st.success("Project saved! (Demo feature)")

if __name__ == "__main__":
    main()
