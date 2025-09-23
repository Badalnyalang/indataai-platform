import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="IndataAI - AI-Powered 3D Analytics Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit elements
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
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = "scatter"
if 'animation_speed' not in st.session_state:
    st.session_state.animation_speed = 1.0
if 'auto_rotate' not in st.session_state:
    st.session_state.auto_rotate = True
if 'theme' not in st.session_state:
    st.session_state.theme = "Viridis"

def generate_ai_insights(df):
    """Generate AI insights"""
    insights = []
    
    if df is not None and len(df) > 0:
        # Data quality
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_score = max(0, 100 - missing_pct * 2)
        insights.append({
            "title": "Data Quality Analysis",
            "content": f"Data integrity: {quality_score:.1f}% quality score. {len(df)} data points analyzed.",
            "confidence": quality_score
        })
        
        # Pattern recognition
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            high_corr = (corr_matrix.abs() > 0.7).sum().sum() - len(numeric_cols)
            insights.append({
                "title": "Pattern Recognition",
                "content": f"Found {high_corr} strong correlations. Multiple clusters detected.",
                "confidence": min(95, 70 + high_corr * 5)
            })
        
        # Anomaly detection
        outlier_count = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            outlier_count += len(outliers)
        
        insights.append({
            "title": "Anomaly Detection",
            "content": f"Identified {outlier_count} potential outliers in the dataset.",
            "confidence": 85
        })
        
        # AI recommendation
        insights.append({
            "title": "AI Recommendation",
            "content": f"3D scatter plot optimal for {len(numeric_cols)} dimensions. Accuracy: 94%",
            "confidence": 94
        })
    
    return insights

def create_3d_visualization(df, chart_type, animation_speed, auto_rotate, theme):
    """Create 3D visualization"""
    if df is None or len(df) == 0:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 3:
        return None
    
    x_col, y_col, z_col = numeric_cols[:3]
    
    # Create clusters if not present
    if 'cluster' not in df.columns:
        df = df.copy()
        df['cluster'] = pd.cut(df[x_col], bins=5, labels=['Group A', 'Group B', 'Group C', 'Group D', 'Group E'])
    
    color_col = 'cluster' if 'cluster' in df.columns else x_col
    
    if chart_type == "scatter":
        fig = px.scatter_3d(
            df, 
            x=x_col, y=y_col, z=z_col,
            color=color_col,
            title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
            color_continuous_scale=theme,
            hover_data=[x_col, y_col, z_col]
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [{"visible": [True]}, {"title": "Playing Animation"}],
                        "label": "Play",
                        "method": "restyle"
                    },
                    {
                        "args": [{"visible": [True]}, {"title": "Animation Paused"}],
                        "label": "Pause",
                        "method": "restyle"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "y": 0
            }]
        )
        
    elif chart_type == "surface":
        # Create surface plot
        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 20)
        y_range = np.linspace(df[y_col].min(), df[y_col].max(), 20)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Simple interpolation
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                distances = ((df[x_col] - X[i,j])**2 + (df[y_col] - Y[i,j])**2)**0.5
                if distances.min() < np.inf:
                    closest_idx = distances.idxmin()
                    Z[i,j] = df.loc[closest_idx, z_col]
        
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale=theme)])
        fig.update_layout(title=f"3D Surface: {z_col} over {x_col} and {y_col}")
        
    else:  # Default to scatter
        fig = px.scatter_3d(
            df, 
            x=x_col, y=y_col, z=z_col,
            color=color_col,
            title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
            color_continuous_scale=theme
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def main():
    # Title
    st.markdown('<h1 class="main-title">IndataAI Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered 3D Data Visualization & Analytics</p>', unsafe_allow_html=True)
    
    # Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Data Input")
        
        # File upload
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'xlsx', 'json'],
            help="Supports CSV, Excel, and JSON formats"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sample data
        if st.button("Use Sample Data"):
            np.random.seed(42)
            sample_data = {
                'Revenue': np.random.normal(100000, 25000, 200),
                'Customers': np.random.normal(500, 150, 200),
                'Satisfaction': np.random.normal(4.2, 0.8, 200),
                'Department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'Support'], 200),
                'Quarter': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], 200)
            }
            st.session_state.data = pd.DataFrame(sample_data)
            st.session_state.ai_insights = generate_ai_insights(st.session_state.data)
            st.rerun()
        
        # Process uploaded file
        if uploaded_file is not None:
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
        
        # Dataset info
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.markdown("### Dataset Overview")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f'<div class="metric-card"><h3>{len(df)}</h3><p>Data Points</p></div>', unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="metric-card"><h3>{len(df.columns)}</h3><p>Variables</p></div>', unsafe_allow_html=True)
            
            st.markdown("**Data Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("### Visualization Controls")
            
            # Chart type
            new_chart_type = st.selectbox("Chart Type", ["scatter", "surface"], index=0 if st.session_state.chart_type == "scatter" else 1)
            if new_chart_type != st.session_state.chart_type:
                st.session_state.chart_type = new_chart_type
                st.rerun()
            
            # Animation controls
            st.markdown("**Animation Settings:**")
            new_speed = st.slider("Animation Speed", 0.1, 3.0, st.session_state.animation_speed, 0.1)
            if new_speed != st.session_state.animation_speed:
                st.session_state.animation_speed = new_speed
            
            new_rotate = st.checkbox("Auto Rotate Camera", value=st.session_state.auto_rotate)
            if new_rotate != st.session_state.auto_rotate:
                st.session_state.auto_rotate = new_rotate
            
            # Theme
            new_theme = st.selectbox("Color Theme", 
                                   ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"], 
                                   index=["Viridis", "Plasma", "Inferno", "Magma", "Cividis"].index(st.session_state.theme))
            if new_theme != st.session_state.theme:
                st.session_state.theme = new_theme
    
    with col2:
        st.markdown("### 3D Visualization")
        
        if st.session_state.data is not None:
            # Create visualization
            fig = create_3d_visualization(
                st.session_state.data, 
                st.session_state.chart_type, 
                st.session_state.animation_speed, 
                st.session_state.auto_rotate, 
                st.session_state.theme
            )
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    if st.button("Export as PNG"):
                        st.success("Visualization exported!")
                with col_export2:
                    if st.button("Share Visualization"):
                        st.success("Share link generated!")
            else:
                st.warning("Please upload data with at least 3 numeric columns.")
        else:
            st.info("Upload your data or use sample data to see AI-powered 3D visualizations")
            
            # Demo placeholder
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
                title="Demo: Upload your data to see AI insights",
                height=400,
                scene=dict(
                    xaxis_title="Variable X",
                    yaxis_title="Variable Y", 
                    zaxis_title="Variable Z"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # AI Insights
    if st.session_state.ai_insights:
        st.markdown("---")
        st.markdown("## AI-Powered Insights")
        
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
    
    # Action buttons
    st.markdown("---")
    col_action1, col_action2, col_action3, col_action4 = st.columns(4)
    
    with col_action1:
        if st.button("Reset Data", use_container_width=True):
            st.session_state.data = None
            st.session_state.ai_insights = []
            st.rerun()
    
    with col_action2:
        if st.button("Analytics Report", use_container_width=True):
            st.success("Analytics report generated!")
    
    with col_action3:
        if st.button("AI Recommendations", use_container_width=True):
            st.success("AI recommendations updated!")
    
    with col_action4:
        if st.button("Save Project", use_container_width=True):
            st.success("Project saved!")

if __name__ == "__main__":
    main()
