# Grompt.py

```python
import argparse
import os
from dotenv import load_dotenv
from pocketgroq import GroqProvider
from pocketgroq.exceptions import GroqAPIKeyMissingError, GroqAPIError

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables or use defaults
DEFAULT_MODEL = os.getenv('GROMPT_DEFAULT_MODEL', 'llama-3.3-70b-versatile')
DEFAULT_TEMPERATURE = float(os.getenv('GROMPT_DEFAULT_TEMPERATURE', '0.5'))
DEFAULT_MAX_TOKENS = int(os.getenv('GROMPT_DEFAULT_MAX_TOKENS', '1024'))

def get_rephrased_user_prompt(user_request: str) -> str:
    """
    Generate a system message for prompt rephrasing.
    
    Args:
        user_request (str): The original user request.
    
    Returns:
        str: A system message for prompt rephrasing.
    """
    return f"""You are a professional prompt engineer. Your task is to optimize the user's prompt by making it clearer, more concise, and more effective. Only output the improved prompt without adding any commentary or labels. If the original prompt is already optimized, return it unchanged. 
    User request: "{user_request}"
    Rephrased:
    """

def rephrase_prompt(prompt: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """
    Rephrase the given prompt using the GroqProvider.
    
    Args:
        prompt (str): The original prompt to rephrase.
        model (str): The model to use for generation.
        temperature (float): The temperature for text generation.
        max_tokens (int): The maximum number of tokens to generate.
    
    Returns:
        str: The rephrased prompt.
    
    Raises:
        GroqAPIKeyMissingError: If the GROQ_API_KEY is not set.
        GroqAPIError: If an error occurs during the API call.
    """
    try:
        groq = GroqProvider()
        
        system_message = get_rephrased_user_prompt(prompt)
        
        response = groq.generate(
            prompt=system_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.strip()
    except GroqAPIKeyMissingError:
        raise GroqAPIKeyMissingError("GROQ_API_KEY must be set in the environment or in a .env file")
    except GroqAPIError as e:
        raise GroqAPIError(f"Error calling Groq API: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Rephrase a user prompt using Groq LLM.")
    parser.add_argument("prompt", help="The user prompt to rephrase.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="The Groq model to use.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="The temperature for text generation.")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="The maximum number of tokens to generate.")
    
    args = parser.parse_args()
    
    try:
        rephrased = rephrase_prompt(args.prompt, args.model, args.temperature, args.max_tokens)
        print("Rephrased prompt:")
        print(rephrased)
    except (GroqAPIKeyMissingError, GroqAPIError) as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def test_function():
    return "Grompt module imported successfully!"

if __name__ == "__main__":
    main()
```

# streamlit_app.py

```python
import streamlit as st
import os
import sys
import importlib.util
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Diagnostic information
# st.write("Current working directory:", os.getcwd())
# st.write("Contents of current directory:", os.listdir())
# st.write("Python path:", sys.path)

# Function to import a module from a file path
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get configuration from environment variables or use defaults
DEFAULT_MODEL = os.getenv('GROMPT_DEFAULT_MODEL', 'llama-3.3-70b-versatile')
DEFAULT_TEMPERATURE = float(os.getenv('GROMPT_DEFAULT_TEMPERATURE', '0.5'))
DEFAULT_MAX_TOKENS = int(os.getenv('GROMPT_DEFAULT_MAX_TOKENS', '1024'))

# Sidebar for API key input and GitHub link
st.sidebar.title("Configuration")
GROQ_API_KEY = st.sidebar.text_input("Enter your GROQ API Key:", type="password")

if not GROQ_API_KEY:
    st.sidebar.warning("Please enter your GROQ API Key to use the app.") 

# Main app
st.title("Grompt - Prompt Optimizer")

st.write("""
Grompt is a utility that uses Groq's LLM services to instantly optimize and rephrase prompts. 
Enter your prompt below and see how Grompt can improve it!  Add it to YOUR project in seconds:
""")
st.write("""<div style='color:grey;'>rephrased = rephrase_prompt("[YOUR PROMPT HERE]")</div> <br/>""", unsafe_allow_html=True)

user_prompt = st.text_area("Enter your prompt:", height=100)

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
    if not GROQ_API_KEY:
        st.error("Please enter your GROQ API Key in the sidebar to use the app.")
    elif user_prompt:
        # Set the API key in the environment for the rephrase_prompt function
        os.environ['GROQ_API_KEY'] = GROQ_API_KEY
        
        # Now import Grompt after setting the API key
        try:
            Grompt = import_module_from_path("Grompt", "Grompt.py")
            # st.write("Successfully imported Grompt")
        except Exception as e:
            st.error(f"Unable to import 'Grompt': {str(e)}")
            st.stop()
        
        with st.spinner("Optimizing your prompt..."):
            optimized_prompt = Grompt.rephrase_prompt(user_prompt, model, temperature, max_tokens)
        if optimized_prompt:
            st.subheader("Optimized Prompt:")
            st.write(optimized_prompt)
    else:
        st.warning("Please enter a prompt to optimize.")

st.markdown("---")
st.write("Powered by Groq LLM services.")

# Add a note about API key security
st.sidebar.markdown("---")
st.sidebar.info(
    "Note: Your API key is used only for this session and is not stored. "
    "Always keep your API keys confidential and do not share them publicly."
)

# Add GitHub link to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("[View on GitHub](https://github.com/jgravelle/Grompt)")

# Add credit to J. Gravelle
st.sidebar.markdown("Created by J. Gravelle")

```

