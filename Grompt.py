"""
Grompt: A Python utility for optimizing prompts using Groq's LLM services.
Developed by Rohith Bollineni
"""

import argparse
import os
import groq
from dotenv import load_dotenv
from typing import Optional
from prompt_canvas import PromptCanvas

load_dotenv()

DEFAULT_MODEL = os.getenv('GROMPT_DEFAULT_MODEL', 'llama-3.3-70b-versatile')
DEFAULT_TEMPERATURE = float(os.getenv('GROMPT_DEFAULT_TEMPERATURE', '0.5'))
DEFAULT_MAX_TOKENS = int(os.getenv('GROMPT_DEFAULT_MAX_TOKENS', '1024'))

def craft_system_message(canvas: Optional[PromptCanvas] = None, prompt: str = "") -> str:
    if canvas:
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
        return get_rephrased_user_prompt(prompt)

def get_rephrased_user_prompt(prompt: str) -> str:
    return f"""You are a professional prompt engineer. Optimize this prompt by making it clearer, more concise, and more effective.
    User request: "{prompt}"
    Rephrased:"""

def rephrase_prompt(prompt: str, 
                   model: str = DEFAULT_MODEL,
                   temperature: float = DEFAULT_TEMPERATURE, 
                   max_tokens: int = DEFAULT_MAX_TOKENS,
                   canvas: Optional[PromptCanvas] = None):
    """Rephrase the given prompt using Groq's LLM."""
    client = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    system_message = craft_system_message(canvas, prompt)
    user_message = get_rephrased_user_prompt(prompt)
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Rephrase prompts using Groq LLM.")
    parser.add_argument("prompt", help="The prompt to rephrase")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    
    args = parser.parse_args()
    
    try:
        rephrased = rephrase_prompt(args.prompt, args.model, args.temperature, args.max_tokens)
        print("Rephrased prompt:")
        print(rephrased)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()