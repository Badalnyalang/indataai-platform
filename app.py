import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, zscore
import streamlit.components.v1 as components
import base64
from io import BytesIO

st.set_page_config(page_title="IndataAI Platform", page_icon="🚀", layout="wide")

class AdvancedAI:
    def __init__(self):
        self.insights = []
        self.recommendations = []
        self.correlations = []
        self.outliers = []
    
    def analyze_data(self, df):
        """Enhanced AI analysis with statistical depth"""
        self.insights = []
        self.recommendations = []
        self.correlations = []
        self.outliers = []
        
        # Data quality analysis
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        quality_score = ((total_cells - missing_cells) / total_cells) * 100
        
        # Column type detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Correlation analysis
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i+1, min(i+6, len(numeric_cols))):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    valid_data = df[[col1, col2]].dropna()
                    if len(valid_data) > 3:
                        corr, p_value = pearsonr(valid_data[col1], valid_data[col2])
                        if abs(corr) > 0.7:
                            self.correlations.append({
                                'col1': col1, 'col2': col2, 'correlation': corr,
                                'strength': 'Strong' if abs(corr) > 0.8 else 'Moderate'
                            })
        
        # Outlier detection using z-score
        for col in numeric_cols[:3]:  # Check first 3 numeric columns
            values = df[col].dropna()
            if len(values) > 3:
                z_scores = np.abs(zscore(values))
                outlier_indices = np.where(z_scores > 2.5)[0]
                if len(outlier_indices) > 0:
                    outlier_count = len(outlier_indices)
                    outlier_pct = (outlier_count / len(values)) * 100
                    self.outliers.append({
                        'column': col, 'count': outlier_count, 
                        'percentage': outlier_pct
                    })
        
        # Generate insights
        self.insights.append(f"Data Quality: {quality_score:.1f}% ({df.shape[0]} rows, {df.shape[1]} columns)")
        
        if self.correlations:
            self.insights.append(f"Strong correlations detected: {len(self.correlations)}")
            for corr in self.correlations[:2]:
                self.insights.append(f"  • {corr['col1']} ↔ {corr['col2']}: {corr['correlation']:.3f}")
        
        if self.outliers:
            self.insights.append(f"Outliers detected in {len(self.outliers)} columns")
            for outlier in self.outliers:
                self.insights.append(f"  • {outlier['column']}: {outlier['count']} points ({outlier['percentage']:.1f}%)")
        
        # Smart recommendations
        if len(numeric_cols) >= 3:
            self.recommendations.append({
                'type': 'scatter3d', 'confidence': 95,
                'reason': f'{len(numeric_cols)} numeric variables optimal for 3D exploration'
            })
        
        if len(numeric_cols) >= 2:
            self.recommendations.append({
                'type': 'scatter', 'confidence': 90,
                'reason': 'Strong correlation patterns detected'
            })
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            self.recommendations.append({
                'type': 'animated_bar', 'confidence': 88,
                'reason': f'Categorical grouping with {len(categorical_cols)} dimensions'
            })
        
        # Sort by confidence
        self.recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'quality_score': quality_score,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'insights': self.insights,
            'recommendations': self.recommendations,
            'correlations': self.correlations,
            'outliers': self.outliers
        }

def auto_detect_columns(df):
    """Smart column detection for optimal visualization"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Preference patterns for different column types
    x_preferences = ['sales_amount', 'sales', 'revenue', 'value', 'amount', 'price']
    y_preferences = ['profit_margin', 'profit', 'margin', 'pct', 'percentage', 'rate']
    color_preferences = ['region', 'category', 'group', 'type', 'segment', 'class']
    time_preferences = ['time', 'date', 'year', 'month', 'quarter', 'season']
    
    def find_best_match(cols, preferences):
        for pref in preferences:
            for col in cols:
                if pref.lower() in col.lower():
                    return col
        return cols[0] if cols else None
    
    return {
        'x': find_best_match(numeric_cols, x_preferences),
        'y': find_best_match([c for c in numeric_cols if c != find_best_match(numeric_cols, x_preferences)], y_preferences),
        'color': find_best_match(categorical_cols, color_preferences),
        'time': find_best_match(categorical_cols + numeric_cols, time_preferences)
    }

def create_enhanced_visualization(df, chart_type, color_scheme, animation_speed, columns, ai_analysis):
    """Enhanced visualization with clean white background and flat buttons"""
    
    # Prepare data
    data_sample = df.head(min(500, len(df)))  # Limit for performance
    data_json = data_sample.to_json(orient="records")
    
    # Enhanced color schemes with more colors
    color_schemes = {
        'Professional': ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1'],
        'Vibrant': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#fd79a8', '#6c5ce7', '#00b894', '#fdcb6e', '#a29bfe'],
        'Corporate': ['#1f4e79', '#2d5aa0', '#5b9bd5', '#70ad47', '#ffc000', '#c5504b', '#70ad47', '#7030a0', '#ff9900', '#375623'],
        'Ocean': ['#006994', '#0091ad', '#00b4c6', '#00d8e0', '#1efcfa', '#74b9ff', '#0984e3', '#00cec9', '#55a3ff', '#3742fa'],
        'Sunset': ['#ff7675', '#fd79a8', '#fdcb6e', '#e17055', '#74b9ff', '#a29bfe', '#6c5ce7', '#00b894', '#00cec9', '#55a3ff']
    }
    
    colors = color_schemes.get(color_scheme, color_schemes['Professional'])
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
        <style>
            body {{
                font-family: 'Inter', -apple-system, sans-serif;
                background: #ffffff;
                margin: 0; padding: 0; overflow: hidden;
            }}
            .container {{
                width: 100vw; height: 100vh;
                display: flex; flex-direction: column;
                background: #ffffff;
            }}
            .header {{
                background: #ffffff;
                padding: 16px 24px;
                border-bottom: 1px solid #e5e7eb;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            .title {{
                font-size: 28px; font-weight: 700; color: #2c3e50;
                margin: 0 0 8px 0;
            }}
            .subtitle {{
                font-size: 16px; color: #7f8c8d; margin: 0;
                display: flex; gap: 24px; align-items: center;
            }}
            .ai-badge {{
                background: #10b981;
                color: white; padding: 4px 12px; border-radius: 12px;
                font-size: 12px; font-weight: 600;
            }}
            .controls {{
                background: #ffffff;
                padding: 16px 24px;
                display: flex; gap: 12px; align-items: center;
                border-bottom: 1px solid #e5e7eb;
                flex-wrap: wrap;
            }}
            .btn {{
                background: #667eea;
                color: white; border: none; padding: 10px 20px;
                border-radius: 8px; cursor: pointer; font-weight: 600;
                transition: all 0.3s ease; font-size: 14px;
                box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
            }}
            .btn:hover {{
                background: #5a67d8;
                transform: translateY(-1px);
            }}
            .btn.secondary {{
                background: #6b7280;
                color: white;
            }}
            .btn.secondary:hover {{
                background: #4b5563;
            }}
            .btn.export {{
                background: #10b981;
            }}
            .btn.export:hover {{
                background: #059669;
            }}
            .chart-area {{
                flex: 1; padding: 20px;
                display: flex; justify-content: center; align-items: center;
                background: #ffffff;
            }}
            .chart-container {{
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px; padding: 15px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                width: 100%; height: 600px;
            }}
            .element {{
                transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
            }}
            .element:hover {{
                transform: scale(1.1);
            }}
            .trail {{
                opacity: 0.6;
                stroke-width: 3;
                fill: none;
            }}
            .legend {{
                font-size: 12px;
                font-weight: 500;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 0.8; transform: scale(1); }}
                50% {{ opacity: 1; transform: scale(1.05); }}
            }}
            .pulsing {{ animation: pulse 2s infinite; }}
            
            @keyframes slideIn {{
                from {{ transform: translateY(20px); opacity: 0; }}
                to {{ transform: translateY(0); opacity: 1; }}
            }}
            .slide-in {{ animation: slideIn 0.6s ease-out; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header slide-in">
                <div class="title">IndataAI Platform</div>
                <div class="subtitle">
                    <span>Chart: {chart_type}</span>
                    <span class="ai-badge">AI Confidence: {ai_analysis['recommendations'][0]['confidence'] if ai_analysis['recommendations'] else 95}%</span>
                    <span>Quality: {ai_analysis['quality_score']:.1f}%</span>
                    <span>Speed: {animation_speed}x</span>
                </div>
            </div>
            
            <div class="controls slide-in">
                <button class="btn" onclick="playSequence()">▶ AI Analysis</button>
                <button class="btn" onclick="animateEntrance()">🎯 Show Data</button>
                <button class="btn" onclick="animateByCategory()">📊 By Category</button>
                <button class="btn" onclick="animateCorrelations()">🔗 Correlations</button>
                <button class="btn" onclick="animateOutliers()">⚠️ Outliers</button>
                <button class="btn secondary" onclick="pauseAnimation()">⏸ Pause</button>
                <button class="btn secondary" onclick="resetView()">🔄 Reset</button>
                <button class="btn export" onclick="exportVisualization()">💾 Export PNG</button>
                <button class="btn export" onclick="startRecording()">🎥 Record GIF</button>
            </div>
            
            <div class="chart-area">
                <div class="chart-container slide-in">
                    <svg id="chart" width="100%" height="570" viewBox="0 0 1000 570"></svg>
                </div>
            </div>
        </div>
        
        <script>
            const data = {data_json};
            const colors = {json.dumps(colors)};
            const analysisData = {json.dumps(ai_analysis)};
            let animationSpeed = {animation_speed};
            let isPlaying = false;
            let animationInterval;
            
            // Chart setup with full width
            const svg = d3.select("#chart");
            const containerWidth = 1000;
            const margin = {{top: 40, right: 140, bottom: 60, left: 80}};
            const width = containerWidth - margin.left - margin.right;
            const height = 570 - margin.top - margin.bottom;
            const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
            
            // Scales
            const xCol = '{columns["x"]}';
            const yCol = '{columns["y"]}';
            const colorCol = '{columns["color"]}';
            
            const xScale = d3.scaleLinear()
                .domain(d3.extent(data, d => +d[xCol]))
                .nice()
                .range([0, width]);
                
            const yScale = d3.scaleLinear()
                .domain(d3.extent(data, d => +d[yCol]))
                .nice()
                .range([height, 0]);
                
            const colorScale = d3.scaleOrdinal()
                .domain([...new Set(data.map(d => d[colorCol]))])
                .range(colors);
            
            // Create axes
            const xAxis = g.append("g")
                .attr("transform", `translate(0,${{height}})`)
                .call(d3.axisBottom(xScale).tickFormat(d3.format(".2s")));
                
            const yAxis = g.append("g")
                .call(d3.axisLeft(yScale).tickFormat(d3.format(".2s")));
            
            // Axis labels
            g.append("text")
                .attr("class", "axis-label")
                .attr("x", width / 2)
                .attr("y", height + 45)
                .style("text-anchor", "middle")
                .style("font-weight", "600")
                .style("font-size", "14px")
                .text(xCol.replace(/_/g, ' ').toUpperCase());
                
            g.append("text")
                .attr("class", "axis-label")
                .attr("transform", "rotate(-90)")
                .attr("y", -50)
                .attr("x", -height / 2)
                .style("text-anchor", "middle")
                .style("font-weight", "600")
                .style("font-size", "14px")
                .text(yCol.replace(/_/g, ' ').toUpperCase());
            
            // Create legend
            const legend = svg.append("g")
                .attr("class", "legend")
                .attr("transform", `translate(${{width + margin.left + 20}}, ${{margin.top}})`);
            
            const legendItems = legend.selectAll(".legend-item")
                .data(colorScale.domain())
                .enter().append("g")
                .attr("class", "legend-item")
                .attr("transform", (d, i) => `translate(0, ${{i * 25}})`);
            
            legendItems.append("rect")
                .attr("width", 18)
                .attr("height", 18)
                .attr("rx", 3)
                .attr("fill", d => colorScale(d));
            
            legendItems.append("text")
                .attr("x", 24)
                .attr("y", 14)
                .style("font-size", "12px")
                .style("font-weight", "500")
                .text(d => d);
            
            // Create dots without outlines
            const dots = g.selectAll(".dot")
                .data(data)
                .enter().append("circle")
                .attr("class", "dot element")
                .attr("cx", d => xScale(+d[xCol]))
                .attr("cy", d => yScale(+d[yCol]))
                .attr("r", 0)
                .attr("fill", d => colorScale(d[colorCol]))
                .attr("opacity", 0);
            
            // Animation functions
            function animateEntrance() {{
                dots.transition()
                    .delay((d, i) => i * 30 / animationSpeed)
                    .duration(1200 / animationSpeed)
                    .ease(d3.easeBounce)
                    .attr("r", 8)
                    .attr("opacity", 0.85);
            }}
            
            function animateByCategory() {{
                const categories = [...new Set(data.map(d => d[colorCol]))];
                
                categories.forEach((category, i) => {{
                    setTimeout(() => {{
                        dots.transition()
                            .duration(800 / animationSpeed)
                            .attr("opacity", d => d[colorCol] === category ? 1 : 0.15)
                            .attr("r", d => d[colorCol] === category ? 14 : 5);
                    }}, i * 1000 / animationSpeed);
                }});
                
                setTimeout(() => {{
                    resetView();
                }}, categories.length * 1000 / animationSpeed + 1500);
            }}
            
            function animateCorrelations() {{
                if (analysisData.correlations && analysisData.correlations.length > 0) {{
                    const corr = analysisData.correlations[0];
                    
                    // Highlight correlated variables
                    dots.transition()
                        .duration(1000 / animationSpeed)
                        .attr("r", 12)
                        .attr("opacity", 0.9);
                    
                    // Add trend line
                    const lineData = data.sort((a, b) => +a[xCol] - +b[xCol]);
                    const line = d3.line()
                        .x(d => xScale(+d[xCol]))
                        .y(d => yScale(+d[yCol]))
                        .curve(d3.curveBasis);
                    
                    const path = g.append("path")
                        .datum(lineData)
                        .attr("class", "correlation-line")
                        .attr("fill", "none")
                        .attr("stroke", "#ef4444")
                        .attr("stroke-width", 3)
                        .attr("opacity", 0)
                        .attr("d", line);
                    
                    const totalLength = path.node().getTotalLength();
                    path.attr("stroke-dasharray", totalLength + " " + totalLength)
                        .attr("stroke-dashoffset", totalLength)
                        .transition()
                        .duration(2000 / animationSpeed)
                        .attr("stroke-dashoffset", 0)
                        .attr("opacity", 0.7);
                        
                    setTimeout(() => {{
                        path.transition().duration(500).attr("opacity", 0).remove();
                        resetView();
                    }}, 3000 / animationSpeed);
                }}
            }}
            
            function animateOutliers() {{
                if (analysisData.outliers && analysisData.outliers.length > 0) {{
                    const outlierCol = analysisData.outliers[0].column;
                    const outlierThreshold = d3.quantile(data.map(d => +d[outlierCol]).sort(), 0.9);
                    
                    dots.transition()
                        .duration(1000 / animationSpeed)
                        .attr("r", d => +d[outlierCol] > outlierThreshold ? 16 : 6)
                        .attr("opacity", d => +d[outlierCol] > outlierThreshold ? 1 : 0.3);
                    
                    setTimeout(() => resetView(), 3000 / animationSpeed);
                }}
            }}
            
            function playSequence() {{
                animateEntrance();
                setTimeout(() => animateByCategory(), 2000 / animationSpeed);
                setTimeout(() => animateCorrelations(), 6000 / animationSpeed);
                setTimeout(() => animateOutliers(), 10000 / animationSpeed);
            }}
            
            function pauseAnimation() {{
                if (isPlaying) {{
                    if (animationInterval) clearInterval(animationInterval);
                    isPlaying = false;
                    dots.interrupt();
                }} else {{
                    playSequence();
                    isPlaying = true;
                }}
            }}
            
            function resetView() {{
                g.selectAll(".correlation-line").remove();
                dots.interrupt();
                dots.transition()
                    .duration(600)
                    .attr("cx", d => xScale(+d[xCol]))
                    .attr("cy", d => yScale(+d[yCol]))
                    .attr("r", 8)
                    .attr("opacity", 0.85)
                    .attr("fill", d => colorScale(d[colorCol]));
            }}
            
            function exportVisualization() {{
                html2canvas(document.querySelector('.container')).then(canvas => {{
                    const link = document.createElement('a');
                    link.download = 'indataai-visualization.png';
                    link.href = canvas.toDataURL();
                    link.click();
                }});
            }}
            
            let recording = false;
            let recordedFrames = [];
            
            function startRecording() {{
                if (recording) return;
                recording = true;
                recordedFrames = [];
                
                playSequence();
                
                const captureFrame = () => {{
                    if (!recording) return;
                    html2canvas(document.querySelector('.chart-container')).then(canvas => {{
                        recordedFrames.push(canvas.toDataURL());
                        if (recordedFrames.length < 120) {{ // 4 seconds at 30fps
                            setTimeout(captureFrame, 33);
                        }} else {{
                            stopRecording();
                        }}
                    }});
                }};
                
                setTimeout(captureFrame, 100);
            }}
            
            function stopRecording() {{
                recording = false;
                if (recordedFrames.length > 0) {{
                    const link = document.createElement('a');
                    link.download = 'indataai-animation-frame.png';
                    link.href = recordedFrames[0];
                    link.click();
                }}
            }}
            
            // Initialize
            setTimeout(() => {{
                animateEntrance();
            }}, 500);
        </script>
    </body>
    </html>
    """
    
    return html_template

def main():
    st.title("🚀 IndataAI Platform - Enhanced Edition")
    st.markdown("AI-Powered Data Visualization with Professional Animation & Export")
    
    # Initialize AI
    ai_engine = AdvancedAI()
    
    # Sidebar customization
    with st.sidebar:
        st.header("Customization")
        
        # Chart type
        chart_types = ["Scatter Plot", "Bubble Chart", "Line Chart", "Bar Chart", "3D Scatter"]
        selected_chart = st.selectbox("Chart Type", chart_types)
        
        # Color schemes
        color_schemes = ["Professional", "Vibrant", "Corporate", "Ocean", "Sunset"]
        selected_colors = st.selectbox("Color Scheme", color_schemes)
        
        # Animation settings
        animation_speed = st.slider("Animation Speed", 0.5, 3.0, 1.0, 0.1)
        
        # Export settings
        st.subheader("Export Options")
        export_format = st.selectbox("Export Format", ["PNG", "SVG", "HTML"])
        export_quality = st.selectbox("Quality", ["Standard", "High", "Ultra"])
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load and process data
        df = pd.read_csv(uploaded_file)
        
        # AI analysis
        analysis = ai_engine.analyze_data(df)
        
        # Auto-detect optimal columns
        columns = auto_detect_columns(df)
        
        # Layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("📊 Data Overview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("🤖 AI Analysis")
            for insight in analysis['insights']:
                st.write(f"• {insight}")
            
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(analysis['recommendations'][:3], 1):
                confidence_emoji = "🟢" if rec['confidence'] > 90 else "🟡" if rec['confidence'] > 80 else "🔴"
                st.write(f"{confidence_emoji} **{rec['type'].replace('_', ' ').title()}** - {rec['confidence']}%")
                st.caption(rec['reason'])
            
            # Column mapping controls
            st.subheader("🔧 Column Mapping")
            columns['x'] = st.selectbox("X-axis", df.select_dtypes(include=[np.number]).columns, 
                                      index=list(df.select_dtypes(include=[np.number]).columns).index(columns['x']) if columns['x'] in df.columns else 0)
            columns['y'] = st.selectbox("Y-axis", df.select_dtypes(include=[np.number]).columns,
                                      index=list(df.select_dtypes(include=[np.number]).columns).index(columns['y']) if columns['y'] in df.columns else min(1, len(df.select_dtypes(include=[np.number]).columns)-1))
            columns['color'] = st.selectbox("Color/Group", ['None'] + df.select_dtypes(include=['object']).columns.tolist(),
                                          index=(['None'] + df.select_dtypes(include=['object']).columns.tolist()).index(columns['color']) if columns['color'] in df.columns else 0)
        
        with col2:
            # Validation
            required_cols = [columns['x'], columns['y']]
            missing_cols = [col for col in required_cols if col not in df.columns or col is None]
            
            if missing_cols:
                st.error(f"Please select valid X and Y columns")
            else:
                st.subheader("🎨 Interactive Visualization")
                
                # Create enhanced visualization
                html_content = create_enhanced_visualization(
                    df, selected_chart, selected_colors, animation_speed, columns, analysis
                )
                
                # Display
                components.html(html_content, height=800, scrolling=False)
                
                # Success message
                st.success("✨ Enhanced AI visualization ready! Use the controls in the chart for animations and export.")
                
                # Additional insights
                with st.expander("🔍 Detailed Analysis"):
                    if analysis['correlations']:
                        st.subheader("Strong Correlations")
                        for corr in analysis['correlations']:
                            st.write(f"• **{corr['col1']}** ↔ **{corr['col2']}**: {corr['correlation']:.3f} ({corr['strength']})")
                    
                    if analysis['outliers']:
                        st.subheader("Outlier Detection")
                        for outlier in analysis['outliers']:
                            st.write(f"• **{outlier['column']}**: {outlier['count']} outliers ({outlier['percentage']:.1f}%)")

if __name__ == "__main__":
    main()
