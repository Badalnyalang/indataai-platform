# indatai_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textwrap import dedent

# Optional interpolation for surface
try:
    from scipy.interpolate import griddata
    SCIPY = True
except Exception:
    SCIPY = False

# Optional image export dependency
KAL = True
try:
    import kaleido  # noqa: F401
except Exception:
    KAL = False

# ---------------------- Page config ----------------------
st.set_page_config(page_title="IndataAI • Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---------------------- Minimal robust CSS for admin look ----------------------
st.markdown(
    dedent(
        """
    <style>
    /* Top header */
    .topbar {
        display:flex; align-items:center; justify-content:space-between;
        padding:10px 18px; border-bottom:1px solid #e6e9ee; background: linear-gradient(90deg,#ffffff,#fbfdff);
    }
    .brand {
        display:flex; align-items:center; gap:12px;
    }
    .brand img { height:34px; border-radius:6px; }
    .brand h1 { font-size:18px; margin:0; color:#0f172a; }
    .brand p { margin:0; font-size:12px; color:#475569; }
    /* Metric cards row */
    .metrics { display:flex; gap:12px; }
    .card { background: #fff; padding:14px; border-radius:8px; box-shadow: 0 1px 3px rgba(12,18,30,0.04); border:1px solid #eef2f7; min-width:140px; }
    .card h3 { margin:0; font-size:20px; color:#0f172a; }
    .card p { margin:0; color:#64748b; font-size:12px; }
    /* insights tile */
    .insight { padding:10px; border-radius:8px; color:white; background: linear-gradient(135deg,#6366f1,#06b6d4); }
    /* compact dataset preview */
    .small { font-size:13px; color:#475569; }
    /* make the main container full width */
    .main .block-container{padding-top:8px;}
    </style>
    """
    ),
    unsafe_allow_html=True,
)

# ---------------------- Session defaults ----------------------
if 'data' not in st.session_state:
    st.session_state.data = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = []
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = "scatter"
if 'theme' not in st.session_state:
    st.session_state.theme = "Viridis"
if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False

# ---------------------- Helpers ----------------------
def generate_ai_insights(df):
    insights = []
    if df is None or df.empty:
        return insights
    # Data quality
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    quality_score = max(0, 100 - missing_pct * 2)
    insights.append({"title": "Data Quality", "content": f"{quality_score:.1f}% completeness", "confidence": quality_score})
    # Numeric analysis
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        strong_pairs = int((upper > 0.7).sum().sum())
        insights.append({"title": "Patterns", "content": f"{strong_pairs} strong numeric correlations (|r|>0.7)", "confidence": min(95, 60 + strong_pairs*5)})
        # outliers unique rows
        outlier_mask = pd.DataFrame(False, index=df.index, columns=num_cols)
        for c in num_cols:
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            mask = (df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)
            outlier_mask[c] = mask
        unique_outliers = int(outlier_mask.any(axis=1).sum())
        insights.append({"title": "Anomalies", "content": f"{unique_outliers} unique rows flagged as outliers (IQR)", "confidence": 85})
    return insights

def normalize_theme(name):
    return {"Viridis":"viridis","Plasma":"plasma","Inferno":"inferno","Magma":"magma","Cividis":"cividis"}.get(name, "viridis")

def create_3d_fig(df, x_col, y_col, z_col, chart_type, theme, color_by=None, animate=False, frames=10):
    theme_norm = normalize_theme(theme)
    if chart_type == "scatter":
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_by if color_by else None,
                            color_continuous_scale=theme_norm if color_by and df[color_by].dtype.kind in 'fiu' else None,
                            hover_data=df.columns.tolist())
    elif chart_type == "animated_scatter":
        # px supports animation_frame — we expect df to have a column 'anim_frame' prepared
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_by if color_by else None,
                            animation_frame='anim_frame',
                            hover_data=df.columns.tolist())
        # reduce frame duration for snappy feel
        fig.layout.updatemenus = [dict(type="buttons",
                                       buttons=[dict(label="Play",
                                                     method="animate",
                                                     args=[None, {"frame": {"duration": 200, "redraw": True},
                                                                  "fromcurrent": True, "transition": {"duration": 0}}])])]
    elif chart_type == "lines":
        # lines connecting points ordered by index or a 'time' column if present
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=df[x_col], y=df[y_col], z=df[z_col], mode='lines+markers', marker=dict(size=3)))
        fig.update_layout(title=f"3D Line: {x_col} / {y_col} / {z_col}")
    elif chart_type == "mesh":
        # Simple mesh using triangulation (works best when points form a surface)
        fig = go.Figure(data=[go.Mesh3d(x=df[x_col], y=df[y_col], z=df[z_col], opacity=0.7)])
        fig.update_layout(title=f"3D Mesh: {x_col}/{y_col}/{z_col}")
    elif chart_type == "surface":
        if not SCIPY:
            st.warning("Surface requires scipy.interpolate.griddata; install scipy or switch to scatter.")
            return None
        xi = np.linspace(df[x_col].min(), df[x_col].max(), 80)
        yi = np.linspace(df[y_col].min(), df[y_col].max(), 80)
        XI, YI = np.meshgrid(xi, yi)
        pts = df[[x_col, y_col]].values
        vals = df[z_col].values
        ZI = griddata(pts, vals, (XI, YI), method='linear')
        fig = go.Figure(data=[go.Surface(x=XI, y=YI, z=ZI, colorscale=theme_norm)])
        fig.update_layout(title=f"Surface: {z_col} over {x_col} & {y_col}")
    else:
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col)

    fig.update_layout(scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col),
                      margin=dict(l=0, r=0, t=45, b=0), height=650)
    return fig

# ---------------------- Header ----------------------
st.markdown(
    dedent(
        """
    <div class="topbar">
        <div class="brand">
            <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Font_Awesome_5_solid_chart-line.svg" />
            <div>
                <h1>IndataAI</h1>
                <p class="small">AI-powered 3D analytics & dashboard</p>
            </div>
        </div>
        <div style="display:flex; gap:12px; align-items:center;">
            <div style="text-align:right;">
                <div style="font-size:12px; color:#64748b;">Signed in as</div>
                <div style="font-weight:600;">You</div>
            </div>
        </div>
    </div>
    """
    ),
    unsafe_allow_html=True,
)

# ---------------------- Sidebar controls ----------------------
with st.sidebar:
    st.header("Controls")
    # Upload
    uploaded = st.file_uploader("Upload dataset (CSV / XLSX / JSON)", type=['csv', 'xlsx', 'json'])
    st.markdown("### Quick actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Sample"):
            # sample dataset
            np.random.seed(42)
            sample = pd.DataFrame({
                "Revenue": np.random.normal(100000, 25000, 500),
                "Customers": np.random.normal(500, 150, 500),
                "Satisfaction": np.random.normal(4.2, 0.8, 500),
                "Department": np.random.choice(['Sales','Marketing','Engineering','Support'], 500),
                "Month": np.tile(np.arange(1,13), 42)[:500]
            })
            st.session_state.data = sample
            st.session_state.sample_loaded = True
            st.session_state.ai_insights = generate_ai_insights(sample)
    with col2:
        if st.button("Clear"):
            st.session_state.data = None
            st.session_state.ai_insights = []
            st.session_state.sample_loaded = False

    st.markdown("---")
    # Chart controls (depends on dataset later)
    st.markdown("### Visualization settings")
    chart_type = st.selectbox("Chart type", options=["scatter", "animated_scatter", "lines", "mesh", "surface"], index=["scatter","animated_scatter","lines","mesh","surface"].index(st.session_state.chart_type) if st.session_state.chart_type in ["scatter","animated_scatter","lines","mesh","surface"] else 0)
    st.session_state.chart_type = chart_type

    theme = st.selectbox("Color theme", ["Viridis","Plasma","Inferno","Magma","Cividis"], index=["Viridis","Plasma","Inferno","Magma","Cividis"].index(st.session_state.theme) if st.session_state.theme in ["Viridis","Plasma","Inferno","Magma","Cividis"] else 0)
    st.session_state.theme = theme

    st.markdown("Animation options (for animated_scatter)")
    frames = st.slider("Frames (auto-binned frames)", min_value=5, max_value=60, value=12, step=1)
    st.markdown("---")
    st.markdown("Export")
    if st.button("Export PNG"):
        # attempt to export last figure if present
        try:
            if "last_fig" in st.session_state and st.session_state.last_fig is not None:
                st.session_state.last_fig.write_image("indatai_visual.png", scale=2)
                st.success("Saved indatai_visual.png on server (app host).")
                if not KAL:
                    st.warning("kaleido not installed on server; saving may fail in some deployments.")
            else:
                st.warning("No visualization to export yet.")
        except Exception as e:
            st.error(f"Export failed: {e}")

# ---------------------- Load uploaded file if present ----------------------
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded)
        elif uploaded.name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded)
        elif uploaded.name.lower().endswith('.json'):
            df = pd.read_json(uploaded)
        else:
            st.error("Unsupported file type")
            df = None

        if df is not None:
            # sample very large files for responsiveness
            max_rows = 300000
            if len(df) > max_rows:
                st.warning(f"Large file ({len(df)} rows). Sampling {max_rows} rows for visualization.")
                df = df.sample(max_rows, random_state=42).reset_index(drop=True)
            st.session_state.data = df
            st.session_state.ai_insights = generate_ai_insights(df)
    except Exception as exc:
        st.error(f"Failed to load uploaded file: {exc}")

# ---------------------- Main content ----------------------
data = st.session_state.data

# Top metrics row
colA, colB, colC, colD = st.columns([1.2,1,1,1])
with colA:
    st.markdown('<div class="card"><h3>{}</h3><p>Rows</p></div>'.format(len(data) if data is not None else 0), unsafe_allow_html=True)
with colB:
    st.markdown('<div class="card"><h3>{}</h3><p>Columns</p></div>'.format(len(data.columns) if data is not None else 0), unsafe_allow_html=True)
with colC:
    num_cols = len(data.select_dtypes(include=[np.number]).columns) if data is not None else 0
    st.markdown('<div class="card"><h3>{}</h3><p>Numeric fields</p></div>'.format(num_cols), unsafe_allow_html=True)
with colD:
    st.markdown('<div class="card"><h3>{}</h3><p>Sample loaded</p></div>'.format("Yes" if st.session_state.sample_loaded else "No"), unsafe_allow_html=True)

st.markdown("")  # spacing

# Two column layout: left = Visualization, right = Insights + Dataset inspector
left, right = st.columns([3, 1.1])

with left:
    st.subheader("3D Visualization")
    if data is None:
        st.info("Upload data or load sample data from the sidebar to begin.")
        # small demo
        demo_df = pd.DataFrame(dict(x=[1,2,3,4,5], y=[2,4,3,5,1], z=[3,1,4,2,5], label=list("ABCDE")))
        demo_fig = px.scatter_3d(demo_df, x='x', y='y', z='z', color='label')
        st.plotly_chart(demo_fig, use_container_width=True)
    else:
        # choose default numeric columns
        numeric = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) < 3:
            st.warning("Please upload data with at least 3 numeric columns for 3D visualizations.")
        else:
            x_col = st.selectbox("X axis", numeric, index=0)
            y_col = st.selectbox("Y axis", numeric, index=1 if len(numeric)>1 else 0)
            z_col = st.selectbox("Z axis", numeric, index=2 if len(numeric)>2 else 0)

            # color option
            color_cols = data.columns.tolist()
            color_by = st.selectbox("Color by (optional)", options=["(none)"]+color_cols, index=0)
            color_by_sel = None if color_by == "(none)" else color_by

            # If animated_scatter selected, prepare frames
            df_plot = data.copy()
            if st.session_state.chart_type == "animated_scatter":
                # Prefer an existing time/frame-like column
                frame_candidates = [c for c in df_plot.columns if 'time' in c.lower() or 'date' in c.lower() or 'frame' in c.lower()]
                if frame_candidates:
                    chosen_frame = frame_candidates[0]
                    df_plot['anim_frame'] = df_plot[chosen_frame].astype(str)
                else:
                    # auto-bin by index or by first numeric column to number of frames from sidebar
                    base_col = numeric[0]
                    df_plot = df_plot.reset_index(drop=True)
                    labels = pd.qcut(df_plot.index.to_series(), q=frames, duplicates='drop', labels=False)
                    df_plot['anim_frame'] = labels.astype(str)
                    st.caption(f"No time/frame column found — auto-binned into {len(df_plot['anim_frame'].unique())} frames based on index.")
            # Create fig
            fig = create_3d_fig(df_plot, x_col, y_col, z_col, st.session_state.chart_type, st.session_state.theme, color_by_sel, animate=(st.session_state.chart_type=="animated_scatter"), frames=frames)
            if fig is None:
                st.error("Could not create the selected chart (surface requires scipy).")
            else:
                st.plotly_chart(fig, use_container_width=True)
                # store for export
                st.session_state.last_fig = fig

                # small action row
                e1, e2, e3 = st.columns([1,1,1])
                with e1:
                    if st.button("Zoom Reset"):
                        st.experimental_rerun()
                with e2:
                    if st.button("Save PNG"):
                        try:
                            st.session_state.last_fig.write_image("indatai_visual.png", scale=2)
                            st.success("Saved indatai_visual.png on server (app host).")
                        except Exception as e:
                            st.error(f"Save failed. Ensure 'kaleido' is available on the host. Error: {e}")
                with e3:
                    if st.button("Open Data Table"):
                        st.experimental_set_query_params(show_table="1")
                        st.experimental_rerun()

with right:
    st.subheader("AI Insights")
    if not st.session_state.ai_insights:
        st.info("AI insights will appear here when data is loaded.")
    else:
        for ins in st.session_state.ai_insights:
            st.markdown(f'<div class="insight" style="padding:10px; margin-bottom:10px;"><strong>{ins["title"]}</strong><div style="font-size:13px;margin-top:6px">{ins["content"]}</div><div style="font-size:11px;margin-top:8px;opacity:0.9">Confidence: {ins["confidence"]:.0f}%</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Dataset Inspector")
    if data is None:
        st.write("No dataset loaded.")
    else:
        st.markdown(f"**Shape:** {data.shape}")
        st.markdown("**Columns:**")
        st.write(list(zip(data.columns, data.dtypes)))
        with st.expander("Preview & stats"):
            st.dataframe(data.head(200), use_container_width=True)
            st.markdown("**Summary (numeric)**")
            st.dataframe(data.describe().T, use_container_width=True)

# ---------------------- Footer / notes ----------------------
st.markdown("---")
st.markdown("Built with ❤️ • IndataAI — admin dashboard prototype. Need a specific admin theme (dark, left nav with icons, or export/sharing to S3)? I can add it next.")
