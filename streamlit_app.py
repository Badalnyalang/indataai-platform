import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="IndataAI - 3D Data Visualization Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to hide Streamlit elements and make it fullscreen
st.markdown("""
<style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    .main .block-container {
        padding-top: 0px;
        padding-bottom: 0px;
        padding-left: 0px;
        padding-right: 0px;
        max-width: 100%;
    }
    iframe {
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

def load_html_component():
    """Load the HTML component from file"""
    try:
        with open('indataai_platform.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        # Fallback HTML content embedded
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>IndataAI Platform</title>
            <style>
                body {
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                    background: #1a1a1a;
                    color: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                }
                .error-message {
                    text-align: center;
                    padding: 40px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    border: 1px solid #333;
                }
            </style>
        </head>
        <body>
            <div class="error-message">
                <h1>🚀 IndataAI Platform</h1>
                <p>Setting up the 3D visualization platform...</p>
                <p>Please ensure indataai_platform.html is in the repository root.</p>
            </div>
        </body>
        </html>
        """

def main():
    # Load and display the HTML component
    html_content = load_html_component()
    
    # Use Streamlit's HTML component to render the full application
    components.html(
        html_content,
        height=800,
        scrolling=False
    )

if __name__ == "__main__":
    main()
