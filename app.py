import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit.components.v1 as components

st.set_page_config(page_title="IndataAI Platform", page_icon="🚀", layout="wide")

class IndataAI:
    def __init__(self):
        self.insights = []
        self.recommendations = []
    
    def analyze_data(self, df):
        self.insights = []
        self.recommendations = []
        
        # Data quality analysis
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_score = 100 - missing_pct
        self.insights.append(f"Data Quality: {quality_score:.1f}%")
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            strong_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i,j]
                    if abs(corr_val) > 0.7:
                        strong_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j], 
                            'value': corr_val
                        })
            
            if strong_correlations:
                self.insights.append(f"Strong correlations: {len(strong_correlations)} found")
        
        # Chart recommendations
        if len(numeric_cols) >= 3:
            self.recommendations.append({
                'type': 'scatter',
                'confidence': 95,
                'reason': 'Multi-dimensional data optimal for scatter analysis'
            })
        
        if len(df.select_dtypes(include=['object']).columns) >= 1:
            self.recommendations.append({
                'type': 'bar',
                'confidence': 90,
                'reason': 'Categorical data suitable for bar charts'
            })
            
        self.recommendations.append({
            'type': 'line',
            'confidence': 85,
            'reason': 'Time-series trends visualization'
        })
        
        return self.insights, self.recommendations

def create_advanced_visualization(df, chart_type, color_scheme, animation_speed, auto_play):
    data_json = df.to_json(orient="records")
    colors = color_scheme
    
    # Dynamic chart selection
    if chart_type == "Scatter Plot":
        chart_js = create_scatter_chart_js(colors, animation_speed, auto_play)
    elif chart_type == "Bar Chart":
        chart_js = create_bar_chart_js(colors, animation_speed, auto_play)
    elif chart_type == "Line Chart":
        chart_js = create_line_chart_js(colors, animation_speed, auto_play)
    else:
        chart_js = create_scatter_chart_js(colors, animation_speed, auto_play)
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0; padding: 20px; min-height: 100vh;
            }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .ai-header {{
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }}
            .ai-title {{ font-size: 24px; font-weight: 700; color: #2c3e50; margin-bottom: 10px; }}
            .ai-subtitle {{ font-size: 16px; color: #7f8c8d; }}
            .controls {{
                background: rgba(255,255,255,0.9);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                text-align: center;
            }}
            .btn {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; border: none; padding: 12px 24px;
                border-radius: 8px; margin: 6px; cursor: pointer;
                font-weight: 600; transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }}
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            }}
            .btn.active {{
                background: linear-gradient(135deg, #00B59C 0%, #00A085 100%);
            }}
            .play-controls {{
                margin-top: 15px;
                display: flex;
                justify-content: center;
                gap: 10px;
            }}
            .chart-container {{
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            }}
            .element {{ transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1); }}
            .element:hover {{ transform: scale(1.05); }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 0.8; }}
                50% {{ opacity: 1; }}
            }}
            .pulsing {{ animation: pulse 2s infinite; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="ai-header">
                <div class="ai-title">IndataAI - Dynamic Analysis</div>
                <div class="ai-subtitle">Chart: {chart_type} | AI Confidence: 95% | Speed: {animation_speed}x</div>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="playAnimation()">▶ Play Animation</button>
                <button class="btn" onclick="pauseAnimation()">⏸ Pause</button>
                <button class="btn" onclick="resetView()">🔄 Reset</button>
                <button class="btn" onclick="analyzeData()">🤖 AI Analyze</button>
                <button class="btn" onclick="showClusters()">🎯 Clusters</button>
                <div class="play-controls">
                    <button class="btn" onclick="changeSpeed(0.5)">0.5x</button>
                    <button class="btn" onclick="changeSpeed(1.0)">1x</button>
                    <button class="btn" onclick="changeSpeed(2.0)">2x</button>
                </div>
            </div>
            
            <div class="chart-container">
                <svg id="chart" width="1000" height="600"></svg>
            </div>
        </div>
        
        <script>
            const data = {data_json};
            const colors = {json.dumps(colors)};
            let animationSpeed = {animation_speed};
            let isPlaying = false;
            let animationInterval;
            
            const svg = d3.select("#chart");
            const margin = {{top: 40, right: 40, bottom: 80, left: 80}};
            const width = 1000 - margin.left - margin.right;
            const height = 600 - margin.top - margin.bottom;
            const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
            
            {chart_js}
            
            function playAnimation() {{
                if (!isPlaying) {{
                    isPlaying = true;
                    startContinuousAnimation();
                }}
            }}
            
            function pauseAnimation() {{
                isPlaying = false;
                if (animationInterval) clearInterval(animationInterval);
            }}
            
            function resetView() {{
                pauseAnimation();
                initializeChart();
            }}
            
            function changeSpeed(speed) {{
                animationSpeed = speed;
                if (isPlaying) {{
                    pauseAnimation();
                    playAnimation();
                }}
            }}
            
            function startContinuousAnimation() {{
                animationInterval = setInterval(() => {{
                    if (isPlaying) {{
                        rotateView();
                        pulseElements();
                    }}
                }}, 2000 / animationSpeed);
            }}
            
            function rotateView() {{
                const angle = Date.now() * 0.001 * animationSpeed;
                g.transition()
                    .duration(1000 / animationSpeed)
                    .attr("transform", `translate(${{margin.left}},${{margin.top}}) rotate(${{angle * 10}}, ${{width/2}}, ${{height/2}})`);
            }}
            
            function pulseElements() {{
                g.selectAll(".element")
                    .transition()
                    .duration(1000 / animationSpeed)
                    .attr("r", d => 12 + Math.sin(Date.now() * 0.01) * 3)
                    .transition()
                    .duration(1000 / animationSpeed)
                    .attr("r", 8);
            }}
            
            // Auto-play if enabled
            {f"playAnimation();" if auto_play else ""}
            
            // Initialize chart
            initializeChart();
        </script>
    </body>
    </html>
    """
    return html_template

def create_scatter_chart_js(colors, animation_speed, auto_play):
    return f"""
        const x = d3.scaleLinear()
            .domain(d3.extent(data, d => d.sales_amount)).nice()
            .range([0, width]);
        const y = d3.scaleLinear()
            .domain(d3.extent(data, d => d.profit_margin)).nice()
            .range([height, 0]);
        const color = d3.scaleOrdinal().range(colors);
        
        function initializeChart() {{
            g.selectAll("*").remove();
            
            // Axes
            g.append("g")
                .attr("transform", `translate(0,${{height}})`)
                .call(d3.axisBottom(x));
            g.append("g").call(d3.axisLeft(y));
            
            // Dots
            g.selectAll(".element")
                .data(data)
                .enter().append("circle")
                .attr("class", "element")
                .attr("cx", d => x(d.sales_amount))
                .attr("cy", d => y(d.profit_margin))
                .attr("r", 0)
                .attr("fill", (d,i) => color(i % colors.length))
                .attr("opacity", 0)
                .transition()
                .delay((d,i) => i * 50 / animationSpeed)
                .duration(1000 / animationSpeed)
                .attr("r", 8)
                .attr("opacity", 0.8);
        }}
        
        function analyzeData() {{
            g.selectAll(".element")
                .transition()
                .duration(1500 / animationSpeed)
                .attr("cy", d => y(d.profit_margin) + (Math.random() - 0.5) * 100)
                .transition()
                .duration(1500 / animationSpeed)
                .attr("cy", d => y(d.profit_margin));
        }}
        
        function showClusters() {{
            const clusters = ["A", "B", "C"];
            g.selectAll(".element")
                .transition()
                .duration(2000 / animationSpeed)
                .attr("fill", () => color(Math.floor(Math.random() * clusters.length)))
                .attr("r", () => 6 + Math.random() * 8);
        }}
    """

def create_bar_chart_js(colors, animation_speed, auto_play):
    return f"""
        let aggregatedData = [];
        if (data.length > 0 && data[0].region) {{
            const regionGroups = d3.group(data, d => d.region);
            aggregatedData = Array.from(regionGroups, ([key, values]) => ({{
                region: key,
                value: d3.mean(values, d => d.sales_amount || 0)
            }}));
        }}
        
        const x = d3.scaleBand().range([0, width]).padding(0.1);
        const y = d3.scaleLinear().range([height, 0]);
        const color = d3.scaleOrdinal().range(colors);
        
        x.domain(aggregatedData.map(d => d.region));
        y.domain([0, d3.max(aggregatedData, d => d.value)]);
        
        function initializeChart() {{
            g.selectAll("*").remove();
            
            // Axes
            g.append("g")
                .attr("transform", `translate(0,${{height}})`)
                .call(d3.axisBottom(x));
            g.append("g").call(d3.axisLeft(y));
            
            // Bars
            g.selectAll(".element")
                .data(aggregatedData)
                .enter().append("rect")
                .attr("class", "element")
                .attr("x", d => x(d.region))
                .attr("width", x.bandwidth())
                .attr("y", height)
                .attr("height", 0)
                .attr("fill", (d,i) => color(i))
                .transition()
                .delay((d,i) => i * 200 / animationSpeed)
                .duration(1000 / animationSpeed)
                .attr("y", d => y(d.value))
                .attr("height", d => height - y(d.value));
        }}
        
        function analyzeData() {{
            g.selectAll(".element")
                .transition()
                .duration(1500 / animationSpeed)
                .attr("height", d => Math.random() * height)
                .transition()
                .duration(1500 / animationSpeed)
                .attr("height", d => height - y(d.value));
        }}
        
        function showClusters() {{
            g.selectAll(".element")
                .transition()
                .duration(2000 / animationSpeed)
                .attr("fill", () => color(Math.floor(Math.random() * colors.length)));
        }}
    """

def create_line_chart_js(colors, animation_speed, auto_play):
    return f"""
        const sortedData = data.sort((a, b) => (a.sales_amount || 0) - (b.sales_amount || 0));
        
        const x = d3.scaleLinear()
            .domain(d3.extent(sortedData, d => d.sales_amount || 0))
            .range([0, width]);
        const y = d3.scaleLinear()
            .domain(d3.extent(sortedData, d => d.profit_margin || 0))
            .range([height, 0]);
        
        const line = d3.line()
            .x(d => x(d.sales_amount || 0))
            .y(d => y(d.profit_margin || 0))
            .curve(d3.curveCardinal);
        
        function initializeChart() {{
            g.selectAll("*").remove();
            
            // Axes
            g.append("g")
                .attr("transform", `translate(0,${{height}})`)
                .call(d3.axisBottom(x));
            g.append("g").call(d3.axisLeft(y));
            
            // Line path
            const path = g.append("path")
                .datum(sortedData)
                .attr("class", "element")
                .attr("fill", "none")
                .attr("stroke", colors[0])
                .attr("stroke-width", 3)
                .attr("d", line);
            
            const totalLength = path.node().getTotalLength();
            path.attr("stroke-dasharray", totalLength + " " + totalLength)
                .attr("stroke-dashoffset", totalLength)
                .transition()
                .duration(3000 / animationSpeed)
                .attr("stroke-dashoffset", 0);
            
            // Points
            g.selectAll(".dot")
                .data(sortedData)
                .enter().append("circle")
                .attr("class", "element dot")
                .attr("cx", d => x(d.sales_amount || 0))
                .attr("cy", d => y(d.profit_margin || 0))
                .attr("r", 0)
                .attr("fill", colors[1])
                .transition()
                .delay((d,i) => i * 100 / animationSpeed)
                .duration(500)
                .attr("r", 5);
        }}
        
        function analyzeData() {{
            g.selectAll(".dot")
                .transition()
                .duration(1500 / animationSpeed)
                .attr("r", 8)
                .attr("fill", () => colors[Math.floor(Math.random() * colors.length)])
                .transition()
                .duration(1500 / animationSpeed)
                .attr("r", 5);
        }}
        
        function showClusters() {{
            g.selectAll(".dot")
                .transition()
                .duration(2000 / animationSpeed)
                .attr("cy", d => y(d.profit_margin || 0) + (Math.random() - 0.5) * 50)
                .transition()
                .duration(2000 / animationSpeed)
                .attr("cy", d => y(d.profit_margin || 0));
        }}
    """

def main():
    st.title("🚀 IndataAI Platform")
    st.markdown("AI-Powered Data Visualization with Advanced Customization")
    
    # Sidebar controls
    st.sidebar.header("Customization Options")
    
    # Chart type selection
    chart_types = ["Scatter Plot", "Bar Chart", "Line Chart"]
    selected_chart = st.sidebar.selectbox("Chart Type", chart_types)
    
    # Color scheme selection
    color_schemes = {
        "Professional": ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
        "Vibrant": ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
        "Corporate": ['#1f4e79', '#2d5aa0', '#5b9bd5', '#a5a5a5', '#70ad47'],
        "Ocean": ['#006994', '#0091ad', '#00b4c6', '#00d8e0', '#1efcfa']
    }
    selected_color_scheme = st.sidebar.selectbox("Color Scheme", list(color_schemes.keys()))
    
    # Animation controls
    animation_speed = st.sidebar.slider("Animation Speed", 0.5, 3.0, 1.0, 0.1)
    auto_play = st.sidebar.checkbox("Auto-play animations")
    
    # Initialize AI
    ai_engine = IndataAI()
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("📊 Data Overview")
            st.dataframe(df.head())
            
            st.subheader("🤖 AI Analysis")
            insights, recommendations = ai_engine.analyze_data(df)
            
            for insight in insights:
                st.write(f"• {insight}")
            
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(recommendations[:3], 1):
                confidence_color = "🟢" if rec['confidence'] > 90 else "🟡" if rec['confidence'] > 80 else "🔴"
                st.write(f"{confidence_color} **{rec['type'].title()}** - {rec['confidence']}%")
        
        with col2:
            required_cols = ['sales_amount', 'profit_margin']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info(f"Available columns: {list(df.columns)}")
            else:
                colors = color_schemes[selected_color_scheme]
                html_content = create_advanced_visualization(
                    df, selected_chart, colors, animation_speed, auto_play
                )
                components.html(html_content, height=800, scrolling=True)
                
                st.success("✨ Interactive visualization ready! Use the controls above the chart.")

if __name__ == "__main__":
    main()
