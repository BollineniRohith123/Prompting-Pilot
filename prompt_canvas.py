"""
PromptCanvas: A structured framework for creating effective prompts.
This module provides a dataclass for organizing prompt components in a structured way.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

@dataclass
class PromptCanvas:
    """
    A structured framework for creating effective prompts.
    
    This class organizes various components of a prompt into a structured format,
    making it easier to create consistent, high-quality prompts for LLMs.
    
    Attributes:
        persona (str): The role or persona the AI should adopt
        audience (str): The target audience for the output
        task (str): The specific task or intent of the prompt
        steps (List[str]): Sequential steps to accomplish the task
        context (str): Background information relevant to the task
        references (List[str]): Sources, links, or references
        output_format (str): Desired format for the output
        tonality (str): Desired tone or style of the output
    """
    persona: str = ""
    audience: str = ""
    task: str = ""
    steps: List[str] = field(default_factory=list)
    context: str = ""
    references: List[str] = field(default_factory=list)
    output_format: str = ""
    tonality: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the PromptCanvas to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the PromptCanvas
        """
        return {k: v if v is not None else [] if k in ['steps', 'references'] else "" 
                for k, v in asdict(self).items()}
    
    def is_valid(self) -> bool:
        """
        Check if the PromptCanvas has the minimum required fields filled.
        
        Returns:
            bool: True if the canvas has at least task field filled, False otherwise
        """
        return bool(self.task.strip())
    
    def get_completion_percentage(self) -> float:
        """
        Calculate the percentage of fields that are filled.
        
        Returns:
            float: Percentage of fields filled (0-100)
        """
        fields = [self.persona, self.audience, self.task, self.context, 
                 self.output_format, self.tonality]
        filled = sum(1 for f in fields if f.strip())
        
        # Count non-empty lists
        if self.steps:
            filled += 1
        if self.references:
            filled += 1
            
        return (filled / 8) * 100
    
    def format_as_text(self) -> str:
        """
        Format the PromptCanvas as a structured text prompt.
        
        Returns:
            str: Formatted text representation of the canvas
        """
        sections = []
        
        if self.persona or self.audience:
            role_section = f"You are a {self.persona}" if self.persona else "You are an expert"
            if self.audience:
                role_section += f" creating content for {self.audience}"
            sections.append(role_section + ".")
        
        if self.task:
            sections.append(f"Task: {self.task}")
        
        if self.steps:
            steps_text = "Steps:\n" + "\n".join(f"- {step}" for step in self.steps)
            sections.append(steps_text)
        
        if self.context:
            sections.append(f"Context: {self.context}")
        
        if self.references:
            refs_text = "References:\n" + "\n".join(f"- {ref}" for ref in self.references)
            sections.append(refs_text)
        
        if self.output_format or self.tonality:
            output_section = "Output Requirements:"
            if self.output_format:
                output_section += f"\n- Format: {self.output_format}"
            if self.tonality:
                output_section += f"\n- Tone: {self.tonality}"
            sections.append(output_section)
        
        return "\n\n".join(sections)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptCanvas':
        """
        Create a PromptCanvas instance from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing PromptCanvas data
            
        Returns:
            PromptCanvas: New PromptCanvas instance
        """
        return cls(
            persona=data.get('persona', ''),
            audience=data.get('audience', ''),
            task=data.get('task', ''),
            steps=data.get('steps', []),
            context=data.get('context', ''),
            references=data.get('references', []),
            output_format=data.get('output_format', ''),
            tonality=data.get('tonality', '')
        )