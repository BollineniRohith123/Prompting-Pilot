"""
Grompt: A Python utility for optimizing prompts using Groq's LLM services.
Developed by Rohith Bollineni

This module provides functions for optimizing prompts using Groq's LLM services.
It can be used as a standalone command-line tool or imported as a module in other Python scripts.
"""

import argparse
import os
import json
import time
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union

import groq
from dotenv import load_dotenv
from prompt_canvas import PromptCanvas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('grompt')

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_MODEL = os.getenv('GROMPT_DEFAULT_MODEL', 'llama-3.3-70b-versatile')
DEFAULT_TEMPERATURE = float(os.getenv('GROMPT_DEFAULT_TEMPERATURE', '0.5'))
DEFAULT_MAX_TOKENS = int(os.getenv('GROMPT_DEFAULT_MAX_TOKENS', '1024'))

# Available models
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma-7b-it",
    "mixtral-8x7b-32768"
]

class GromptError(Exception):
    """Base exception class for Grompt errors."""
    pass

class APIKeyError(GromptError):
    """Exception raised for API key issues."""
    pass

class ModelError(GromptError):
    """Exception raised for model-related issues."""
    pass

class PromptError(GromptError):
    """Exception raised for prompt-related issues."""
    pass

def validate_api_key(api_key: Optional[str] = None) -> bool:
    """
    Validate the Groq API key.
    
    Args:
        api_key (Optional[str]): The API key to validate. If None, uses the environment variable.
        
    Returns:
        bool: True if the API key is valid, False otherwise.
        
    Raises:
        APIKeyError: If no API key is provided or found in environment variables.
    """
    key = api_key or os.getenv('GROQ_API_KEY')
    
    if not key:
        raise APIKeyError("No Groq API key provided. Set the GROQ_API_KEY environment variable or pass it as an argument.")
    
    # Basic validation - Groq API keys typically start with 'gsk_'
    if not key.startswith('gsk_'):
        logger.warning("API key doesn't match expected format (should start with 'gsk_')")
        return False
        
    # Further validation would require making an actual API call
    return True

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    This is a rough estimate based on word count and punctuation.
    
    Args:
        text (str): The text to estimate tokens for.
        
    Returns:
        int: Estimated number of tokens.
    """
    # Split by whitespace
    words = text.split()
    
    # Count punctuation that might be tokenized separately
    punctuation_count = len(re.findall(r'[.,!?;:()[\]{}"\'-]', text))
    
    # Estimate: words + punctuation with some adjustment factor
    return len(words) + punctuation_count

def get_system_prompt_for_optimization() -> str:
    """
    Get the system prompt for prompt optimization.
    
    Returns:
        str: The system prompt.
    """
    return """You are an expert prompt engineer with deep knowledge of how to craft effective prompts for large language models.
Your task is to optimize user prompts to make them clearer, more specific, and more effective.

When optimizing prompts, follow these principles:
1. Clarity: Ensure the prompt is clear and unambiguous
2. Specificity: Add specific details and requirements
3. Structure: Organize the prompt logically with clear sections
4. Context: Include relevant context that helps the model understand the task
5. Constraints: Define any constraints or limitations
6. Output format: Specify the desired output format
7. Examples: Include examples if helpful

Your optimized prompts should be comprehensive yet concise, providing all necessary information without unnecessary verbosity."""

def craft_system_message(canvas: Optional[PromptCanvas] = None, prompt: str = "") -> str:
    """
    Craft a system message for the LLM based on the provided canvas or prompt.
    
    Args:
        canvas (Optional[PromptCanvas]): The prompt canvas to use.
        prompt (str): The raw prompt to use if no canvas is provided.
        
    Returns:
        str: The crafted system message.
    """
    if canvas:
        # Use the format_as_text method if available, otherwise fall back to manual formatting
        if hasattr(canvas, 'format_as_text'):
            return canvas.format_as_text()
        else:
            return f"""You are a {canvas.persona} focused on delivering results for {canvas.audience}.

Task: {canvas.task}

Step-by-Step Approach:
{chr(10).join(f'- {step}' for step in canvas.steps)}

Context: {canvas.context}

References: {', '.join(canvas.references)}

Output Requirements:
- Format: {canvas.output_format}
- Tone: {canvas.tonality}"""
    else:
        return get_system_prompt_for_optimization()

def get_rephrased_user_prompt(prompt: str) -> str:
    """
    Format the user prompt for optimization.
    
    Args:
        prompt (str): The original prompt to optimize.
        
    Returns:
        str: The formatted user message.
    """
    return f"""Please optimize this prompt to make it clearer, more specific, and more effective:

ORIGINAL PROMPT:
"{prompt}"

OPTIMIZED PROMPT:"""

def rephrase_prompt(
    prompt: str, 
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE, 
    max_tokens: int = DEFAULT_MAX_TOKENS,
    canvas: Optional[PromptCanvas] = None,
    api_key: Optional[str] = None,
    stream: bool = False
) -> Union[str, None]:
    """
    Rephrase the given prompt using Groq's LLM.
    
    Args:
        prompt (str): The prompt to rephrase.
        model (str): The Groq model to use.
        temperature (float): The temperature for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        canvas (Optional[PromptCanvas]): The prompt canvas to use.
        api_key (Optional[str]): The Groq API key to use. If None, uses the environment variable.
        stream (bool): Whether to stream the response.
        
    Returns:
        Union[str, None]: The rephrased prompt, or None if an error occurred.
        
    Raises:
        APIKeyError: If the API key is invalid.
        ModelError: If the model is invalid.
        PromptError: If the prompt is invalid.
    """
    # Validate inputs
    if not prompt:
        raise PromptError("Prompt cannot be empty")
    
    if model not in AVAILABLE_MODELS:
        logger.warning(f"Model '{model}' not in list of known models: {', '.join(AVAILABLE_MODELS)}")
    
    # Get API key
    key = api_key or os.getenv('GROQ_API_KEY')
    if not validate_api_key(key):
        raise APIKeyError("Invalid Groq API key")
    
    # Initialize client
    client = groq.Groq(api_key=key)
    
    # Prepare messages
    system_message = craft_system_message(canvas, prompt)
    user_message = get_rephrased_user_prompt(prompt)
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    # Log request details (excluding API key)
    logger.info(f"Sending request to Groq API with model={model}, temperature={temperature}, max_tokens={max_tokens}")
    logger.debug(f"System message: {system_message[:100]}...")
    logger.debug(f"User message: {user_message[:100]}...")
    
    start_time = time.time()
    
    try:
        # Make API request
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            # For streaming, return a generator
            def stream_generator():
                collected_content = []
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        collected_content.append(content)
                        yield content
                
                # Return the full content at the end
                return "".join(collected_content)
            
            return stream_generator()
        else:
            # For non-streaming, return the content directly
            result = completion.choices[0].message.content
            
            # Log completion time
            elapsed_time = time.time() - start_time
            logger.info(f"Request completed in {elapsed_time:.2f} seconds")
            
            return result
            
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        raise GromptError(f"Error calling Groq API: {str(e)}")

def save_prompt_history(
    original_prompt: str,
    optimized_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    canvas: Optional[PromptCanvas] = None,
    file_path: Optional[str] = None
) -> str:
    """
    Save prompt history to a JSON file.
    
    Args:
        original_prompt (str): The original prompt.
        optimized_prompt (str): The optimized prompt.
        model (str): The model used.
        temperature (float): The temperature used.
        max_tokens (int): The max tokens used.
        canvas (Optional[PromptCanvas]): The prompt canvas used.
        file_path (Optional[str]): The file path to save to. If None, uses a default path.
        
    Returns:
        str: The file path where the history was saved.
    """
    history_item = {
        "timestamp": datetime.now().isoformat(),
        "original_prompt": original_prompt,
        "optimized_prompt": optimized_prompt,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "canvas": canvas.to_dict() if canvas else None
    }
    
    # Default file path in user's home directory
    if not file_path:
        home_dir = os.path.expanduser("~")
        history_dir = os.path.join(home_dir, ".grompt")
        os.makedirs(history_dir, exist_ok=True)
        file_path = os.path.join(history_dir, "prompt_history.json")
    
    # Load existing history if file exists
    history = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse history file {file_path}, creating new file")
    
    # Add new item and save
    history.append(history_item)
    
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Saved prompt history to {file_path}")
    return file_path

def load_prompt_history(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load prompt history from a JSON file.
    
    Args:
        file_path (Optional[str]): The file path to load from. If None, uses a default path.
        
    Returns:
        List[Dict[str, Any]]: The loaded history.
    """
    # Default file path in user's home directory
    if not file_path:
        home_dir = os.path.expanduser("~")
        file_path = os.path.join(home_dir, ".grompt", "prompt_history.json")
    
    if not os.path.exists(file_path):
        logger.warning(f"History file {file_path} does not exist")
        return []
    
    try:
        with open(file_path, 'r') as f:
            history = json.load(f)
        logger.info(f"Loaded {len(history)} history items from {file_path}")
        return history
    except json.JSONDecodeError:
        logger.error(f"Could not parse history file {file_path}")
        return []

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Rephrase prompts using Groq LLM.")
    parser.add_argument("prompt", help="The prompt to rephrase")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=AVAILABLE_MODELS,
                        help=f"The model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Temperature for text generation (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--api-key", help="Groq API key (overrides environment variable)")
    parser.add_argument("--save-history", action="store_true", help="Save prompt history")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Validate API key
        api_key = args.api_key or os.getenv('GROQ_API_KEY')
        if not validate_api_key(api_key):
            print("Error: Invalid Groq API key")
            return 1
        
        # Estimate tokens
        estimated_tokens = estimate_tokens(args.prompt)
        print(f"Estimated tokens in prompt: {estimated_tokens}")
        
        # Rephrase prompt
        print("Optimizing prompt...")
        start_time = time.time()
        
        rephrased = rephrase_prompt(
            args.prompt,
            args.model,
            args.temperature,
            args.max_tokens,
            api_key=args.api_key
        )
        
        elapsed_time = time.time() - start_time
        
        if rephrased:
            print(f"\nOptimized prompt (in {elapsed_time:.2f} seconds):")
            print("-" * 80)
            print(rephrased)
            print("-" * 80)
            
            # Save history if requested
            if args.save_history:
                save_prompt_history(
                    args.prompt,
                    rephrased,
                    args.model,
                    args.temperature,
                    args.max_tokens
                )
                print("Prompt history saved.")
            
            return 0
        else:
            print("Error: Failed to optimize prompt")
            return 1
            
    except GromptError as e:
        print(f"Error: {str(e)}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        logger.exception("Unexpected error")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)