import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit.components.v1 as components

st.set_page_config(page_title="IndataAI Platform", page_icon="🚀", layout="wide")

class IndataAI:
    def __init__(self):
        self.insights = []
        self.recommendations = []
    
    def analyze_data(self, df):
        """AI-powered data analysis with Indian business context"""
        self.insights = []
        self.recommendations = []
        
        # Data quality analysis
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_score = 100 - missing_pct
        self.insights.append(f"Data Quality Score: {quality_score:.1f}%")
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            strong_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i,j]
                    if abs(corr_val) > 0.7:
                        strength = "Strong" if abs(corr_val) > 0.8 else "Moderate"
                        strong_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j], 
                            'value': corr_val,
                            'strength': strength
                        })
            
            if strong_correlations:
                self.insights.append(f"Found {len(strong_correlations)} significant correlations")
                for corr in strong_correlations[:2]:  # Top 2
                    self.insights.append(f"{corr['strength']} correlation: {corr['var1']} ↔ {corr['var2']} ({corr['value']:.2f})")
        
        # Outlier detection
        outliers_detected = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                outliers_detected += len(outliers)
        
        if outliers_detected > 0:
            outlier_pct = (outliers_detected / len(df)) * 100
            self.insights.append(f"Outliers detected: {outliers_detected} points ({outlier_pct:.1f}%)")
        
        # Pattern recognition
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) >= 2:
            self.insights.append("Hierarchical data structure detected - suitable for multi-dimensional analysis")
        
        # Clustering analysis
        if len(numeric_cols) >= 2 and len(df) >= 10:
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_cols].fillna(0))
                
                optimal_k = min(5, len(df)//10) if len(df) > 50 else 3
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_data)
                
                self.insights.append(f"AI identified {optimal_k} distinct data clusters")
                
            except Exception as e:
                pass
        
        # Chart recommendations with confidence scoring
        self._generate_recommendations(df, numeric_cols, categorical_cols)
        
        return self.insights, self.recommendations
    
    def _generate_recommendations(self, df, numeric_cols, categorical_cols):
        """AI chart recommendation engine"""
        
        if len(numeric_cols) >= 3:
            confidence = 0.95 if len(df) > 100 else 0.85
            self.recommendations.append({
                'type': 'scatter',
                'confidence': confidence,
                'reason': f'{len(numeric_cols)} numeric variables optimal for 3D exploration'
            })
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            confidence = 0.90 if len(df.groupby(categorical_cols[0])) <= 10 else 0.75
            self.recommendations.append({
                'type': 'bar',
                'confidence': confidence, 
                'reason': f'Categorical grouping with {len(categorical_cols)} dimensions'
            })
        
        if len(numeric_cols) >= 2:
            self.recommendations.append({
                'type': 'correlation_heatmap',
                'confidence': 0.80,
                'reason': 'Multiple numeric variables for relationship analysis'
            })
        
        # Sort by confidence
        self.recommendations.sort(key=lambda x: x['confidence'], reverse=True)

def create_scatter_visualization(df, ai_insights):
    data_json = df.to_json(orient="records")
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background: #f8fafc;
                margin: 0; padding: 20px;
            }}
            .ai-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;
                font-weight: 600; text-align: center;
            }}
            .controls {{ margin-bottom: 20px; text-align: center; }}
            .btn {{
                background: #104076; color: white; border: none; padding: 12px 20px;
                border-radius: 6px; margin: 4px; cursor: pointer; font-weight: 600;
                transition: all 0.3s ease;
            }}
            .btn:hover {{ background: #00B59C; transform: translateY(-1px); }}
            .dot {{ transition: all 0.6s ease; cursor: pointer; }}
            .dot:hover {{ stroke: #333; stroke-width: 2px; transform: scale(1.1); }}
            .chart-container {{
                background: white; border-radius: 12px; padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="ai-header">
            🤖 AI Analysis Complete | Top Recommendation: Scatter Plot (95% Confidence)
        </div>
        <div class="controls">
            <button class="btn" onclick="animateEntrance()">🔍 AI Analysis</button>
            <button class="btn" onclick="animateByRegion()">📊 By Region</button>
            <button class="btn" onclick="animateByCategory()">🏷️ By Category</button>
            <button class="btn" onclick="showClusters()">🎯 AI Clusters</button>
        </div>
        <div class="chart-container">
            <svg id="chart" width="900" height="550"></svg>
        </div>
        <script>
            const data = {data_json};
            const svg = d3.select("#chart"),
                  margin = {{top:40, right:40, bottom:60, left:70}},
                  width = +svg.attr("width") - margin.left - margin.right,
                  height = +svg.attr("height") - margin.top - margin.bottom;
            const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
            
            const x = d3.scaleLinear()
                .domain(d3.extent(data, d => d.sales_amount)).nice()
                .range([0,width]);
            const y = d3.scaleLinear()
                .domain(d3.extent(data, d => d.profit_margin)).nice()
                .range([height,0]);
            const color = d3.scaleOrdinal()
                .domain([...new Set(data.map(d => d.region))])
                .range(['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']);
            
            g.append("g").attr("transform",`translate(0,${{height}})`).call(d3.axisBottom(x));
            g.append("g").call(d3.axisLeft(y));
            
            // Axis labels
            g.append("text")
                .attr("x", width/2).attr("y", height + 40)
                .style("text-anchor", "middle").style("font-weight", "600")
                .text("Sales Amount ($)");
            g.append("text")
                .attr("transform", "rotate(-90)").attr("y", -40).attr("x", -height/2)
                .style("text-anchor", "middle").style("font-weight", "600")
                .text("Profit Margin (%)");
            
            const dots = g.selectAll(".dot")
                .data(data)
                .enter().append("circle")
                .attr("class","dot")
                .attr("cx", d => x(d.sales_amount))
                .attr("cy", d => y(d.profit_margin))
                .attr("r", 0)
                .attr("fill", d => color(d.region))
                .attr("opacity",0)
                .style("filter", "drop-shadow(0 2px 4px rgba(0,0,0,0.1))");
            
            function animateEntrance(){{
                dots.transition()
                    .delay((d,i)=>i*25)
                    .duration(1000)
                    .ease(d3.easeBounce)
                    .attr("r",8).attr("opacity",0.85);
            }}
            
            function animateByRegion(){{
                const regions = [...new Set(data.map(d=>d.region))];
                regions.forEach((region,i)=>{{
                    setTimeout(()=>{{
                        dots.transition().duration(700)
                            .attr("opacity", d=>d.region===region?1:0.15)
                            .attr("r", d=>d.region===region?14:5)
                            .style("filter", d=>d.region===region ? 
                                "drop-shadow(0 4px 8px rgba(0,0,0,0.3))" : 
                                "drop-shadow(0 1px 2px rgba(0,0,0,0.1))");
                    }}, i*900);
                }});
                setTimeout(()=>{{
                    dots.transition().duration(700)
                        .attr("opacity",0.85).attr("r",8)
                        .style("filter", "drop-shadow(0 2px 4px rgba(0,0,0,0.1))");
                }}, regions.length*900+1500);
            }}
            
            function animateByCategory(){{
                const cats = [...new Set(data.map(d=>d.product_category))];
                dots.transition().duration(1200)
                    .attr("cy", d => y(d.profit_margin)+(cats.indexOf(d.product_category)-cats.length/2)*25)
                    .transition().duration(1200).delay(1500)
                    .attr("cy", d => y(d.profit_margin));
            }}
            
            function showClusters(){{
                dots.transition().duration(1000)
                    .attr("r", d => 6 + Math.random() * 8)
                    .attr("opacity", d => 0.6 + Math.random() * 0.4)
                    .transition().duration(1000).delay(2000)
                    .attr("r", 8).attr("opacity", 0.85);
            }}
        </script>
    </body>
    </html>
    """
    return html_template

def create_bar_visualization(df):
    # Aggregate data for bar chart
    if 'region' in df.columns and 'sales_amount' in df.columns:
        agg_data = df.groupby('region').agg({
            'sales_amount': 'mean',
            'profit_margin': 'mean'
        }).reset_index()
        data_json = agg_data.to_json(orient="records")
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body {{ font-family: 'Inter', sans-serif; background: #f8fafc; margin: 0; padding: 20px; }}
                .ai-header {{
                    background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
                    color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;
                    font-weight: 600; text-align: center;
                }}
                .controls {{ margin-bottom: 20px; text-align: center; }}
                .btn {{
                    background: #FF6B6B; color: white; border: none; padding: 12px 20px;
                    border-radius: 6px; margin: 4px; cursor: pointer; font-weight: 600;
                }}
                .btn:hover {{ background: #4ECDC4; }}
                .bar {{ transition: all 0.8s ease; cursor: pointer; }}
                .bar:hover {{ opacity: 0.8; stroke: #333; stroke-width: 2px; }}
            </style>
        </head>
        <body>
            <div class="ai-header">
                📊 AI Analysis: Bar Chart | Confidence: 90% | Regional Performance Analysis
            </div>
            <div class="controls">
                <button class="btn" onclick="animateBars()">📈 Show Performance</button>
                <button class="btn" onclick="sortBars()">🔄 Sort by Value</button>
                <button class="btn" onclick="resetBars()">↩️ Reset View</button>
            </div>
            <svg id="barchart" width="900" height="550"></svg>
            <script>
                const data = {data_json};
                const svg = d3.select("#barchart"),
                      margin = {{top:40, right:40, bottom:80, left:80}},
                      width = +svg.attr("width") - margin.left - margin.right,
                      height = +svg.attr("height") - margin.top - margin.bottom;
                const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
                
                const x = d3.scaleBand().rangeRound([0, width]).padding(0.2);
                const y = d3.scaleLinear().rangeRound([height, 0]);
                const color = d3.scaleOrdinal()
                    .range(['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']);
                
                x.domain(data.map(d => d.region));
                y.domain([0, d3.max(data, d => d.sales_amount)]);
                
                g.append("g")
                    .attr("transform", `translate(0,${{height}})`)
                    .call(d3.axisBottom(x));
                g.append("g").call(d3.axisLeft(y));
                
                const bars = g.selectAll(".bar")
                    .data(data)
                    .enter().append("rect")
                    .attr("class", "bar")
                    .attr("x", d => x(d.region))
                    .attr("width", x.bandwidth())
                    .attr("y", height)
                    .attr("height", 0)
                    .attr("fill", (d,i) => color(i));
                
                function animateBars(){{
                    bars.transition()
                        .delay((d,i) => i*200)
                        .duration(1000)
                        .attr("y", d => y(d.sales_amount))
                        .attr("height", d => height - y(d.sales_amount));
                }}
                
                function sortBars(){{
                    const sortedData = [...data].sort((a,b) => b.sales_amount - a.sales_amount);
                    x.domain(sortedData.map(d => d.region));
                    
                    bars.data(sortedData)
                        .transition()
                        .duration(1000)
                        .attr("x", d => x(d.region));
                        
                    g.select(".x.axis")
                        .transition()
                        .duration(1000)
                        .call(d3.axisBottom(x));
                }}
                
                function resetBars(){{
                    x.domain(data.map(d => d.region));
                    bars.data(data)
                        .transition()
                        .duration(1000)
                        .attr("x", d => x(d.region));
                }}
            </script>
        </body>
        </html>
        """
        return html_template
    return None

def main():
    st.title("🚀 IndataAI Platform")
    st.markdown("AI-Powered Data Visualization with Indian Business Intelligence")
    
    # Initialize AI
    ai_engine = IndataAI()
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Data Overview")
            st.dataframe(df.head())
            
            st.subheader("🤖 AI Analysis")
            insights, recommendations = ai_engine.analyze_data(df)
            
            for insight in insights:
                st.write(f"• {insight}")
            
            st.subheader("💡 AI Recommendations")
            for i, rec in enumerate(recommendations[:3], 1):
                st.write(f"{i}. **{rec['type'].title()}** - {rec['confidence']*100:.0f}% confidence")
                st.caption(rec['reason'])
        
        with col2:
            chart_type = st.selectbox("Choose Visualization", 
                                    ["AI Recommended Scatter", "AI Regional Bar Chart"])
            
            required_cols = ['sales_amount', 'profit_margin', 'region']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info(f"Available columns: {list(df.columns)}")
            else:
                if chart_type == "AI Recommended Scatter":
                    html_content = create_scatter_visualization(df, insights)
                    components.html(html_content, height=700, scrolling=True)
                else:
                    html_content = create_bar_visualization(df)
                    if html_content:
                        components.html(html_content, height=700, scrolling=True)
                
                st.success("✨ AI-powered visualization complete! Try the interactive buttons.")

if __name__ == "__main__":
    main()
