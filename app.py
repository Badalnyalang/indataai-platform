import streamlit as st
import pandas as pd
import json
from io import StringIO
import streamlit.components.v1 as components

st.set_page_config(page_title="IndataAI Platform", page_icon="🚀", layout="wide")

def create_d3_visualization(df):
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
                background: #f5f7fa;
                margin: 0; padding: 20px;
            }}
            .controls {{ margin-bottom: 20px; }}
            .btn {{
                background: #104076; color: white; border: none; padding: 10px 18px;
                border-radius: 6px; margin: 4px; cursor: pointer; font-weight: 600;
            }}
            .btn:hover {{ background: #00B59C; }}
            .dot {{ transition: all 0.6s; cursor: pointer; }}
            .dot:hover {{ stroke: #333; stroke-width: 2px; }}
        </style>
    </head>
    <body>
        <div class="controls">
            <button class="btn" onclick="animateEntrance()">Analyze Data</button>
            <button class="btn" onclick="animateByRegion()">By Region</button>
            <button class="btn" onclick="animateByCategory()">By Category</button>
        </div>
        <svg id="chart" width="800" height="500"></svg>
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
            const color = d3.scaleOrdinal(d3.schemeCategory10);
            
            g.append("g").attr("transform",`translate(0,${{height}})`).call(d3.axisBottom(x));
            g.append("g").call(d3.axisLeft(y));
            
            const dots = g.selectAll(".dot")
                .data(data)
                .enter().append("circle")
                .attr("class","dot")
                .attr("cx", d => x(d.sales_amount))
                .attr("cy", d => y(d.profit_margin))
                .attr("r", 0)
                .attr("fill", d => color(d.region))
                .attr("opacity",0);
            
            function animateEntrance(){{
                dots.transition()
                    .delay((d,i)=>i*20)
                    .duration(800)
                    .attr("r",7).attr("opacity",0.85);
            }}
            
            function animateByRegion(){{
                const regions = [...new Set(data.map(d=>d.region))];
                regions.forEach((region,i)=>{{
                    setTimeout(()=>{{
                        dots.transition().duration(600)
                            .attr("opacity", d=>d.region===region?1:0.15)
                            .attr("r", d=>d.region===region?12:4);
                    }}, i*800);
                }});
                setTimeout(()=>{{
                    dots.transition().duration(600)
                        .attr("opacity",0.85).attr("r",7);
                }}, regions.length*800+1000);
            }}
            
            function animateByCategory(){{
                const cats = [...new Set(data.map(d=>d.product_category))];
                dots.transition().duration(1000)
                    .attr("cy", d => y(d.profit_margin)+(cats.indexOf(d.product_category)-cats.length/2)*20)
                    .transition().duration(1000).delay(1200)
                    .attr("cy", d => y(d.profit_margin));
            }}
        </script>
    </body>
    </html>
    """
    return html_template

def main():
    st.title("🚀 IndataAI Platform")
    st.markdown("AI-Powered Data Visualization with Flourish-style Animations")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        required_cols = ['sales_amount', 'profit_margin', 'region']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info(f"Available columns: {list(df.columns)}")
        else:
            st.subheader("Interactive 3D Visualization")
            html_content = create_d3_visualization(df)
            components.html(html_content, height=600, scrolling=True)
            
            st.success("Click the buttons above the chart to see different animations!")

if __name__ == "__main__":
    main()
