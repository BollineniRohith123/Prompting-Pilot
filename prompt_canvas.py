from dataclasses import dataclass
from typing import List

@dataclass
class PromptCanvas:
    persona: str
    audience: str
    task: str
    steps: List[str]
    context: str
    references: List[str]
    output_format: str
    tonality: str