import streamlit as st
import os
import sys
import importlib.util
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import altair as alt

# Load custom CSS
def load_css(file_path):
    with open(file_path, "r") as f:
        return f.read()

# Apply CSS based on theme
def apply_theme_css(theme):
    css_path = os.path.join(os.path.dirname(__file__), "src", "style.css")
    if os.path.exists(css_path):
        css = load_css(css_path)
        
        # Add theme class
        if theme != "Light":
            theme_class = f"{theme.lower()}-theme"
            css += f"\n.stApp {{ {theme_class} }}\n"
        
        return css
    else:
        # Fallback CSS if file doesn't exist
        return """
        .main-header {
            font-size: 2.5rem;
            color: #4527A0;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #5E35B1;
            margin-bottom: 1rem;
        }
        """

# Set page configuration
st.set_page_config(
    page_title="Grompt - Prompt Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

@dataclass
class PromptCanvas:
    persona: str = ""
    audience: str = ""
    task: str = ""
    steps: List[str] = None
    context: str = ""
    references: List[str] = None
    output_format: str = ""
    tonality: str = ""
    
    def to_dict(self):
        return {k: v if v is not None else [] if k in ['steps', 'references'] else "" 
                for k, v in asdict(self).items()}

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'current_optimized' not in st.session_state:
    st.session_state.current_optimized = ""
if 'canvas' not in st.session_state:
    st.session_state.canvas = PromptCanvas()
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {
        "llama-3.3-70b-versatile": {"count": 0, "avg_time": 0},
        "llama3-groq-8b-8192-tool-use-preview": {"count": 0, "avg_time": 0},
        "llama3-70b-8192": {"count": 0, "avg_time": 0},
        "llama3-8b-8192": {"count": 0, "avg_time": 0}
    }

# Apply CSS
css = apply_theme_css(st.session_state.theme)
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Environment variables
DEFAULT_MODEL = os.getenv('GROMPT_DEFAULT_MODEL', 'llama-3.3-70b-versatile')
DEFAULT_TEMPERATURE = float(os.getenv('GROMPT_DEFAULT_TEMPERATURE', '0.5'))
DEFAULT_MAX_TOKENS = int(os.getenv('GROMPT_DEFAULT_MAX_TOKENS', '1024'))
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Sidebar configuration
st.sidebar.markdown('<div class="sub-header">⚙️ Configuration</div>', unsafe_allow_html=True)

# Theme selector
theme_options = ["Light", "Dark", "Blue", "Green", "Purple"]
selected_theme = st.sidebar.selectbox("Select Theme", theme_options, index=theme_options.index(st.session_state.theme))

if selected_theme != st.session_state.theme:
    st.session_state.theme = selected_theme
    # Apply updated theme CSS
    css = apply_theme_css(st.session_state.theme)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.experimental_rerun()

# API Key validation
if not GROQ_API_KEY:
    api_key = st.sidebar.text_input("Enter your GROQ API Key:", type="password")
    if api_key:
        os.environ['GROQ_API_KEY'] = api_key
        GROQ_API_KEY = api_key
        st.sidebar.success("API Key set successfully!")
    else:
        st.error("⚠️ Please set your GROQ_API_KEY in the .env file or enter it in the sidebar.")
        st.stop()

# Main content
st.markdown('<div class="main-header">🧠 Grompt - Prompt Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Grompt uses Groq\'s LLM services to instantly optimize your prompts for better results. Choose between Basic and Advanced modes to craft the perfect prompt.</div>', unsafe_allow_html=True)

# Add tabs for different modes
tab1, tab2, tab3, tab4 = st.tabs(["🔄 Basic", "🎨 Advanced (Prompt Canvas)", "📊 Analytics", "📜 History"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Basic Prompt Optimization</div>', unsafe_allow_html=True)
    
    user_prompt = st.text_area(
        "Enter your prompt:",
        height=150,
        help="Type your original prompt here. Grompt will optimize it for better results with LLMs.",
        placeholder="e.g., Write a story about a robot"
    )
    
    if user_prompt:
        st.session_state.current_prompt = user_prompt
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Prompt Canvas</div>', unsafe_allow_html=True)
    st.markdown("Use the Prompt Canvas to create a highly structured and effective prompt.")
    
    with st.expander("Persona & Audience", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            persona = st.text_input(
                "Persona/Role",
                value=st.session_state.canvas.persona,
                placeholder="e.g., expert technical writer",
                help="Define the role or persona the AI should adopt"
            )
        with col2:
            audience = st.text_input(
                "Target Audience",
                value=st.session_state.canvas.audience,
                placeholder="e.g., software developers",
                help="Specify who the output is intended for"
            )
    
    task = st.text_area(
        "Task/Intent",
        value=st.session_state.canvas.task,
        placeholder="Describe the specific task...",
        help="Clearly state what you want the AI to accomplish"
    )
    
    steps_text = "\n".join(st.session_state.canvas.steps) if st.session_state.canvas.steps else ""
    steps = st.text_area(
        "Steps",
        value=steps_text,
        placeholder="Enter steps, one per line...",
        help="Break down the task into sequential steps"
    )
    
    context = st.text_area(
        "Context",
        value=st.session_state.canvas.context,
        placeholder="Provide relevant background...",
        help="Add any background information that helps frame the task"
    )
    
    references_text = "\n".join(st.session_state.canvas.references) if st.session_state.canvas.references else ""
    references = st.text_area(
        "References",
        value=references_text,
        placeholder="Enter references, one per line...",
        help="Include any sources, links, or references"
    )
    
    with st.expander("Output Format & Tone", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            output_format = st.selectbox(
                "Output Format",
                ["Natural Text", "Technical Documentation", "Code", "Markdown", "JSON", "HTML", "CSV", "Table"],
                index=0 if not st.session_state.canvas.output_format else 
                      ["Natural Text", "Technical Documentation", "Code", "Markdown", "JSON", "HTML", "CSV", "Table"].index(st.session_state.canvas.output_format),
                help="Select the desired format for the output"
            )
        with col2:
            tonality = st.text_input(
                "Tone",
                value=st.session_state.canvas.tonality,
                placeholder="e.g., professional, technical",
                help="Specify the tone or style of the output"
            )
    
    # Update session state canvas
    st.session_state.canvas = PromptCanvas(
        persona=persona,
        audience=audience,
        task=task,
        steps=[s.strip() for s in steps.split('\n') if s.strip()],
        context=context,
        references=[r.strip() for r in references.split('\n') if r.strip()],
        output_format=output_format,
        tonality=tonality
    )
    
    canvas_prompt = st.text_area(
        "Your Prompt:",
        height=150,
        placeholder="Enter your prompt to optimize with the canvas settings above...",
        help="This is the base prompt that will be enhanced with your canvas settings"
    )
    
    if canvas_prompt:
        st.session_state.current_prompt = canvas_prompt
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Usage Analytics</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        # Create metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Prompts Optimized", len(st.session_state.history))
        with col2:
            avg_tokens = sum(h.get("max_tokens", 0) for h in st.session_state.history) / len(st.session_state.history)
            st.metric("Average Tokens", f"{int(avg_tokens)}")
        with col3:
            avg_temp = sum(h.get("temperature", 0) for h in st.session_state.history) / len(st.session_state.history)
            st.metric("Average Temperature", f"{avg_temp:.2f}")
        
        # Model usage chart
        st.subheader("Model Usage")
        model_counts = {}
        for h in st.session_state.history:
            model = h.get("model", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1
        
        model_df = pd.DataFrame({
            "Model": list(model_counts.keys()),
            "Count": list(model_counts.values())
        })
        
        chart = alt.Chart(model_df).mark_bar().encode(
            x=alt.X('Model', sort='-y'),
            y='Count',
            color=alt.Color('Model', scale=alt.Scale(scheme='category10'))
        ).properties(
            width=600,
            height=300
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Time series of usage
        st.subheader("Usage Over Time")
        dates = [datetime.fromisoformat(h.get("timestamp", datetime.now().isoformat())) for h in st.session_state.history]
        date_counts = {}
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            date_counts[date_str] = date_counts.get(date_str, 0) + 1
        
        time_df = pd.DataFrame({
            "Date": list(date_counts.keys()),
            "Count": list(date_counts.values())
        })
        
        time_chart = alt.Chart(time_df).mark_line(point=True).encode(
            x='Date',
            y='Count',
        ).properties(
            width=600,
            height=300
        )
        
        st.altair_chart(time_chart, use_container_width=True)
    else:
        st.info("No analytics data available yet. Start optimizing prompts to see analytics.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Prompt History</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Prompt {len(st.session_state.history) - i}: {item.get('timestamp', 'Unknown date')}"):
                st.markdown("**Original Prompt:**")
                st.text(item.get("original_prompt", ""))
                st.markdown("**Optimized Prompt:**")
                st.text(item.get("optimized_prompt", ""))
                st.markdown("**Settings:**")
                st.text(f"Model: {item.get('model', 'unknown')}, Temperature: {item.get('temperature', 0)}, Max Tokens: {item.get('max_tokens', 0)}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Reuse Original #{len(st.session_state.history) - i}"):
                        st.session_state.current_prompt = item.get("original_prompt", "")
                        st.experimental_rerun()
                with col2:
                    if st.button(f"Reuse Optimized #{len(st.session_state.history) - i}"):
                        st.session_state.current_prompt = item.get("optimized_prompt", "")
                        st.experimental_rerun()
    else:
        st.info("No history available yet. Start optimizing prompts to build history.")
    
    # Export/Import history
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export History") and st.session_state.history:
            history_json = json.dumps(st.session_state.history, indent=2)
            st.download_button(
                label="Download History JSON",
                data=history_json,
                file_name="grompt_history.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("Import History", type="json")
        if uploaded_file is not None:
            try:
                imported_history = json.load(uploaded_file)
                if isinstance(imported_history, list):
                    st.session_state.history.extend(imported_history)
                    st.success(f"Successfully imported {len(imported_history)} history items!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid history format. Please upload a valid JSON file.")
            except Exception as e:
                st.error(f"Error importing history: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Shared model settings
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Model Settings</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    model = st.selectbox("Select Model", [
        "llama-3.3-70b-versatile",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "gemma-7b-it",
        "mixtral-8x7b-32768"
    ], index=0, help="Choose the Groq LLM model to use for optimization")
with col2:
    temperature = st.slider(
        "Temperature",
        0.0, 1.0, DEFAULT_TEMPERATURE, 0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
with col3:
    max_tokens = st.number_input(
        "Max Tokens",
        1, 32768, DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens in the generated response"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Optimize button
optimize_col1, optimize_col2 = st.columns([3, 1])
with optimize_col2:
    optimize_button = st.button("🚀 Optimize Prompt", use_container_width=True)

# Process optimization
if optimize_button:
    current_prompt = st.session_state.current_prompt
    if current_prompt:
        os.environ['GROQ_API_KEY'] = GROQ_API_KEY
        
        try:
            Grompt = import_module_from_path("Grompt", "Grompt.py")
        except Exception as e:
            st.error(f"Unable to import 'Grompt': {str(e)}")
            st.stop()
        
        with st.spinner("🔄 Optimizing your prompt..."):
            # Add a progress bar for visual feedback
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            start_time = time.time()
            
            if tab2._active:  # Advanced mode
                optimized_prompt = Grompt.rephrase_prompt(
                    prompt=current_prompt, 
                    model=model, 
                    temperature=temperature, 
                    max_tokens=max_tokens, 
                    canvas=st.session_state.canvas
                )
            else:  # Basic mode
                optimized_prompt = Grompt.rephrase_prompt(
                    prompt=current_prompt, 
                    model=model, 
                    temperature=temperature, 
                    max_tokens=max_tokens
                )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update model metrics
            if model in st.session_state.model_metrics:
                metrics = st.session_state.model_metrics[model]
                metrics["count"] += 1
                metrics["avg_time"] = ((metrics["avg_time"] * (metrics["count"] - 1)) + processing_time) / metrics["count"]
            
            if optimized_prompt:
                st.session_state.current_optimized = optimized_prompt
                
                # Add to history
                history_item = {
                    "timestamp": datetime.now().isoformat(),
                    "original_prompt": current_prompt,
                    "optimized_prompt": optimized_prompt,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "processing_time": processing_time,
                    "canvas": st.session_state.canvas.to_dict() if tab2._active else None
                }
                st.session_state.history.append(history_item)
                
                # Show success message with processing time
                st.success(f"✅ Prompt optimized successfully in {processing_time:.2f} seconds!")
                
                # Display comparison
                st.markdown('<div class="sub-header">Prompt Comparison</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="comparison-column" style="background-color: #f0f0f0;">', unsafe_allow_html=True)
                    st.markdown("### Original Prompt")
                    st.text_area("", value=current_prompt, height=200, disabled=True, key="original_display")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="comparison-column" style="background-color: #e0f7fa;">', unsafe_allow_html=True)
                    st.markdown("### Optimized Prompt")
                    st.text_area("", value=optimized_prompt, height=200, disabled=True, key="optimized_display")
                    
                    # Copy button
                    if st.button("📋 Copy to Clipboard"):
                        st.code(optimized_prompt)
                        st.success("Copied to clipboard! (Use Ctrl+C to copy the code above)")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show token estimation
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Prompt Analysis:**")
                original_tokens = len(current_prompt.split())
                optimized_tokens = len(optimized_prompt.split())
                token_change = optimized_tokens - original_tokens
                token_percent = (token_change / original_tokens) * 100 if original_tokens > 0 else 0
                
                st.markdown(f"Original prompt: ~{original_tokens} tokens")
                st.markdown(f"Optimized prompt: ~{optimized_tokens} tokens")
                st.markdown(f"Change: {token_change:+d} tokens ({token_percent:+.1f}%)")
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.error("❌ Failed to generate optimized prompt. Please try again.")
    else:
        st.warning("⚠️ Please enter a prompt to optimize.")

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Powered by Rohith Bollineni and Groq's LLM services.")
st.markdown("© 2023 Grompt - All rights reserved")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Note: Your API key is used only for this session and is not stored. "
    "Always keep your API keys confidential."
)

# Display model metrics in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
for model_name, metrics in st.session_state.model_metrics.items():
    if metrics["count"] > 0:
        st.sidebar.markdown(f"**{model_name}**")
        st.sidebar.markdown(f"Uses: {metrics['count']}")
        st.sidebar.markdown(f"Avg. Time: {metrics['avg_time']:.2f}s")

st.sidebar.markdown("---")
st.sidebar.markdown("Created by Rohith Bollineni")