import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Optional: scipy for clean surface interpolation; if not available we fallback
try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ---- Page config ----
st.set_page_config(
    page_title="IndataAI - AI-Powered 3D Analytics Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- CSS (keep minimal & robust) ----
st.markdown("""
<style>
/* Keep CSS minimal to reduce breakage across Streamlit versions */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 100%;
}
.main-title {font-size: 2.2rem; font-weight: 700; background: linear-gradient(90deg,#6366f1,#06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom:0.25rem;}
.subtitle {font-size: 1.05rem; color:#64748b; margin-bottom:1.25rem;}
.ai-insight {background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:0.9rem; border-radius:10px; color: white; border-left:4px solid #06b6d4;}
.metric-card {background:#f8fafc; padding:0.75rem; border-radius:8px; border:1px solid #e2e8f0; text-align:center;}
.upload-section {border:2px dashed #06b6d4; border-radius:10px; padding:1.25rem; text-align:center; background:#f0f9ff; margin:0.75rem 0;}
</style>
""", unsafe_allow_html=True)

# ---- Session state defaults ----
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
    st.session_state.theme = "viridis"

# ---- Helper functions ----
def generate_ai_insights(df):
    insights = []
    if df is None or df.empty:
        return insights

    # Data quality
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    quality_score = max(0, 100 - missing_pct * 2)
    insights.append({
        "title": "Data Quality Analysis",
        "content": f"Data integrity: {quality_score:.1f}% quality score. {len(df)} rows and {len(df.columns)} columns analyzed.",
        "confidence": quality_score
    })

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        # Count unique strong correlations (upper triangle, excluding diagonal)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        strong_pairs = (upper > 0.7).sum().sum()
        insights.append({
            "title": "Pattern Recognition",
            "content": f"Found {int(strong_pairs)} strongly correlated numeric variable pairs (|r| > 0.7).",
            "confidence": min(95, 60 + int(strong_pairs) * 5)
        })

        # Outliers: count unique rows that are outlier in any numeric column (IQR method)
        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            mask = (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
            outlier_mask[col] = mask
        unique_outlier_rows = outlier_mask.any(axis=1).sum()
        insights.append({
            "title": "Anomaly Detection",
            "content": f"Identified {int(unique_outlier_rows)} unique rows with outlier values (IQR method).",
            "confidence": 85
        })

        insights.append({
            "title": "AI Recommendation",
            "content": f"3D scatter is recommended when you have 3+ numerical dimensions. Found {len(numeric_cols)} numeric columns.",
            "confidence": 92
        })

    return insights

def _normalize_theme(name):
    # Plotly accepts lowercase names for built-in continuous scales
    mapping = {
        "Viridis": "viridis",
        "Plasma": "plasma",
        "Inferno": "inferno",
        "Magma": "magma",
        "Cividis": "cividis"
    }
    return mapping.get(name, "viridis")

def create_3d_visualization(df, chart_type, animation_speed, auto_rotate, theme):
    if df is None or df.empty:
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        return None

    x_col, y_col, z_col = numeric_cols[:3]
    df_local = df.copy()

    # If no cluster column, create a simple categorical bucketing for color
    if 'cluster' not in df_local.columns:
        df_local['cluster'] = pd.cut(df_local[x_col], bins=5, labels=['A','B','C','D','E'])

    theme_norm = _normalize_theme(theme)

    if chart_type == "scatter":
        # If color column is categorical -> use discrete coloring
        if df_local['cluster'].dtype.name in ['category', 'object'] or df_local['cluster'].dtype == 'bool':
            fig = px.scatter_3d(
                df_local,
                x=x_col, y=y_col, z=z_col,
                color='cluster',
                title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}",
                hover_data=[x_col, y_col, z_col]
            )
        else:
            fig = px.scatter_3d(
                df_local,
                x=x_col, y=y_col, z=z_col,
                color=z_col,
                title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}",
                color_continuous_scale=theme_norm,
                hover_data=[x_col, y_col, z_col]
            )

    elif chart_type == "surface":
        # Surface: attempt robust interpolation using scipy if available
        if not SCIPY_AVAILABLE:
            # fallback: notify and return None
            st.warning("scipy not available — surface interpolation requires scipy. Install scipy or switch to scatter.")
            return None

        # build grid
        xi = np.linspace(df_local[x_col].min(), df_local[x_col].max(), 60)
        yi = np.linspace(df_local[y_col].min(), df_local[y_col].max(), 60)
        XI, YI = np.meshgrid(xi, yi)
        # Use griddata interpolation
        points = df_local[[x_col, y_col]].values
        values = df_local[z_col].values
        ZI = griddata(points, values, (XI, YI), method='linear')

        fig = go.Figure(data=[go.Surface(x=XI, y=YI, z=ZI, colorscale=theme_norm)])
        fig.update_layout(title=f"3D Surface: {z_col} over {x_col} & {y_col}")

    else:
        # default to scatter
        fig = px.scatter_3d(
            df_local,
            x=x_col, y=y_col, z=z_col,
            color='cluster',
            title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}"
        )

    # layout & camera
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        )
    )

    # NOTE: Plotly camera auto-rotation requires JS; Streamlit plotly chart won't auto-rotate without embedding JS.
    # We still expose animation_speed and auto_rotate for future extension or custom components.

    return fig

# ---- Main UI ----
def main():
    st.markdown('<h1 class="main-title">IndataAI Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered 3D Data Visualization & Analytics</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Data Input")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your dataset (CSV / XLSX / JSON)", type=['csv', 'xlsx', 'json'])
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Use Sample Data"):
            np.random.seed(42)
            sample_data = {
                'Revenue': np.random.normal(100000, 25000, 200),
                'Customers': np.random.normal(500, 150, 200),
                'Satisfaction': np.random.normal(4.2, 0.8, 200),
                'Department': np.random.choice(['Sales','Marketing','Engineering','Support'], 200),
                'Quarter': np.random.choice(['Q1','Q2','Q3','Q4'], 200)
            }
            st.session_state.data = pd.DataFrame(sample_data)
            st.session_state.ai_insights = generate_ai_insights(st.session_state.data)
            st.experimental_rerun()

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.lower().endswith('.json'):
                    df = pd.read_json(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    df = None

                if df is not None:
                    # If dataset is very large, sample to keep UI responsive
                    max_rows = 200000
                    if len(df) > max_rows:
                        st.warning(f"Dataset has {len(df)} rows. Sampling {max_rows} rows for visualization.")
                        df = df.sample(max_rows, random_state=42).reset_index(drop=True)

                    st.session_state.data = df
                    st.session_state.ai_insights = generate_ai_insights(st.session_state.data)
                    st.success(f"Data loaded: {len(st.session_state.data)} rows, {len(st.session_state.data.columns)} columns.")
            except Exception as e:
                st.error(f"Error loading file: {e}")

        # Dataset info preview
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
            new_chart_type = st.selectbox("Chart Type", ["scatter", "surface"], index=0 if st.session_state.chart_type=="scatter" else 1)
            if new_chart_type != st.session_state.chart_type:
                st.session_state.chart_type = new_chart_type

            # Animation controls (informational; actual auto-rotation requires custom components)
            st.markdown("**Camera / Animation Settings (informational)**")
            st.session_state.animation_speed = st.slider("Animation Speed (informational)", 0.1, 3.0, st.session_state.animation_speed, 0.1)
            st.session_state.auto_rotate = st.checkbox("Auto Rotate Camera (requires custom component)", value=st.session_state.auto_rotate)

            # Theme
            themes = ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"]
            new_theme = st.selectbox("Color Theme", themes, index=themes.index(st.session_state.theme.title()) if st.session_state.theme.title() in themes else 0)
            if new_theme != st.session_state.theme:
                st.session_state.theme = new_theme

    with col2:
        st.markdown("### 3D Visualization")
        if st.session_state.data is not None:
            fig = create_3d_visualization(
                st.session_state.data,
                st.session_state.chart_type,
                st.session_state.animation_speed,
                st.session_state.auto_rotate,
                st.session_state.theme
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    if st.button("Export as PNG"):
                        try:
                            # requires kaleido on server to save images: pip install -U kaleido
                            fig.write_image("visualization.png", scale=2)
                            st.success("Saved visualization.png to server (download from app host).")
                        except Exception as e:
                            st.error("Export failed. Server needs 'kaleido' or image export capability. " + str(e))
                with col_export2:
                    if st.button("Share Visualization"):
                        st.info("Sharing not implemented. Implement backend saving + permalink generation.")
            else:
                st.warning("Please upload data with at least 3 numeric columns (or enable scipy for surface).")
        else:
            st.info("Upload your data or use sample data to see AI-powered 3D visualizations")
            # demo scatter
            demo_fig = go.Figure()
            demo_fig.add_trace(go.Scatter3d(
                x=[1,2,3,4,5],
                y=[2,4,3,5,1],
                z=[3,1,4,2,5],
                mode='markers',
                marker=dict(size=7),
                text=['A','B','C','D','E']
            ))
            demo_fig.update_layout(title="Demo: upload data to see AI insights", height=400,
                                   scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
            st.plotly_chart(demo_fig, use_container_width=True)

    # AI Insights display
    if st.session_state.ai_insights:
        st.markdown("---")
        st.markdown("## AI-Powered Insights")
        cols = st.columns(len(st.session_state.ai_insights))
        for i, ins in enumerate(st.session_state.ai_insights):
            with cols[i]:
                st.markdown(f"""
                <div class="ai-insight">
                    <h4>{ins['title']}</h4>
                    <p>{ins['content']}</p>
                    <div style="margin-top:0.5rem;">
                        <small>Confidence: {ins['confidence']:.1f}%</small>
                        <div style="background: rgba(255,255,255,0.2); height:4px; border-radius:2px; margin-top:6px;">
                            <div style="background:white; height:4px; width:{ins['confidence']}%; border-radius:2px;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Actions
    st.markdown("---")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button("Reset Data"):
            st.session_state.data = None
            st.session_state.ai_insights = []
            st.experimental_rerun()
    with a2:
        if st.button("Analytics Report"):
            st.success("Analytics report stub (implement export pipeline).")
    with a3:
        if st.button("AI Recommendations"):
            st.success("AI recommendations refreshed.")
    with a4:
        if st.button("Save Project"):
            st.success("Project saved (stub).")

if __name__ == "__main__":
    main()
