import streamlit as st
import os
import sys
import importlib.util
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List

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

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

DEFAULT_MODEL = os.getenv('GROMPT_DEFAULT_MODEL', 'llama-3.3-70b-versatile')
DEFAULT_TEMPERATURE = float(os.getenv('GROMPT_DEFAULT_TEMPERATURE', '0.5'))
DEFAULT_MAX_TOKENS = int(os.getenv('GROMPT_DEFAULT_MAX_TOKENS', '1024'))
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in the .env file")
    st.stop()

st.sidebar.title("Configuration")

st.title("Grompt - Prompt Optimizer")
st.write("Grompt uses Groq's LLM services to instantly optimize prompts.")
st.write("Developed by Rohith Bollineni")

# Add tabs for Basic and Advanced modes
tab1, tab2 = st.tabs(["Basic", "Advanced (Prompt Canvas)"])

with tab1:
    user_prompt = st.text_area("Enter your prompt:", height=100)

with tab2:
    with st.expander("Persona & Audience", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            persona = st.text_input("Persona/Role", placeholder="e.g., expert technical writer")
        with col2:
            audience = st.text_input("Target Audience", placeholder="e.g., software developers")
    
    task = st.text_area("Task/Intent", placeholder="Describe the specific task...")
    steps = st.text_area("Steps", placeholder="Enter steps, one per line...")
    context = st.text_area("Context", placeholder="Provide relevant background...")
    references = st.text_area("References", placeholder="Enter references, one per line...")
    
    with st.expander("Output Format & Tone", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            output_format = st.selectbox("Output Format", 
                ["Natural Text", "Technical Documentation", "Code", "Markdown"])
        with col2:
            tonality = st.text_input("Tone", placeholder="e.g., professional, technical")
    
    canvas_prompt = st.text_area("Your Prompt:", height=100)

# Shared model settings
col1, col2, col3 = st.columns(3)
with col1:
    model = st.selectbox("Select Model", [
        "llama-3.3-70b-versatile",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama3-70b-8192",
        "llama3-8b-8192"
    ], index=0)
with col2:
    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)
with col3:
    max_tokens = st.number_input("Max Tokens", 1, 32768, DEFAULT_MAX_TOKENS)

if st.button("Optimize Prompt"):
    if user_prompt or canvas_prompt:
        os.environ['GROQ_API_KEY'] = GROQ_API_KEY
        
        try:
            Grompt = import_module_from_path("Grompt", "Grompt.py")
        except Exception as e:
            st.error(f"Unable to import 'Grompt': {str(e)}")
            st.stop()
        
        with st.spinner("Optimizing prompt..."):
            if canvas_prompt:  # Advanced mode
                canvas = PromptCanvas(
                    persona=persona,
                    audience=audience,
                    task=task,
                    steps=[s.strip() for s in steps.split('\n') if s.strip()],
                    context=context,
                    references=[r.strip() for r in references.split('\n') if r.strip()],
                    output_format=output_format,
                    tonality=tonality
                )
                optimized_prompt = Grompt.rephrase_prompt(
                    prompt=canvas_prompt, 
                    model=model, 
                    temperature=temperature, 
                    max_tokens=max_tokens, 
                    canvas=canvas
                )
            else:  # Basic mode
                optimized_prompt = Grompt.rephrase_prompt(
                    prompt=user_prompt, 
                    model=model, 
                    temperature=temperature, 
                    max_tokens=max_tokens
                )
                
            if optimized_prompt:
                st.write("### Optimized Prompt:")
                st.write(optimized_prompt)
            else:
                st.error("Failed to generate optimized prompt. Please try again.")
    else:
        st.warning("Please enter a prompt to optimize.")

st.markdown("---")
st.write("Powered by Rohith Bollineni and Groq's LLM services.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Note: Your API key is used only for this session and is not stored. "
    "Always keep your API keys confidential."
)

st.sidebar.markdown("---")
st.sidebar.markdown("Created by Rohith Bollineni")