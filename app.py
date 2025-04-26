# app.py
import streamlit as st
from main import run_research
import time

# Configure page
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="centered"
)

# Sidebar for API configuration
with st.sidebar:
    st.title("🔑 API Configuration")
    tavily_key = st.text_input("Tavily API Key", type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    search_depth = st.selectbox("Search Depth", ["Basic", "Advanced"], index=1)

# Main interface
st.title("🔍 AI Research Agent System")
query = st.text_area("Enter your research query:", 
                    height=150,
                    placeholder="e.g. 'What are the latest developments in AI agent architectures?'")

if st.button("Start Research"):
    if not all([tavily_key, anthropic_key]):
        st.error("Please provide both API keys in the sidebar!")
        st.stop()
    
    if not query:
        st.error("Please enter a research query!")
        st.stop()

    # Set API keys as environment variables
    import os
    os.environ["TAVILY_API_KEY"] = tavily_key
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    # Initialize progress
    progress_bar = st.progress(0)
    status = st.empty()
    
    try:
        # Simulated progress stages
        stages = [
            ("Initializing research agent...", 10),
            ("Searching web resources...", 25),
            ("Analyzing findings...", 45),
            ("Synthesizing answer...", 75),
            ("Finalizing report...", 100)
        ]

        result = None
        with st.spinner("Processing your request..."):
            # Run research system
            result = run_research(query)
            
            # Update progress
            for stage_text, progress in stages:
                progress_bar.progress(progress)
                status.text(f"Status: {stage_text}")
                time.sleep(1)  # Simulate processing time

        # Display results
        st.success("✅ Research Complete!")
        st.subheader("Research Report")
        st.markdown(result)
        
        # Add download option
        st.download_button(
            label="Download Report",
            data=result,
            file_name="research_report.md",
            mime="text/markdown"
        )

    except Exception as e:
        st.error(f"Research failed: {str(e)}")
