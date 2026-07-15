from typing import List, Optional
from pydantic import BaseModel, Field


class FactCheckedAnswer(BaseModel):
    """The final structured answer along with reasoning and sourced facts."""
    reasoning: str = Field(
        description="Step-by-step logical reasoning path used to answer the question."
    )
    extracted_facts: List[str] = Field(
        description="List of specific critical claims or facts used to make this conclusion."
    )
    main_subject: str = Field(
        description="The primary core answer (e.g., the capital city name)."
    )
    requested_metrics: Optional[str] = Field(
        description="Any figures, population numbers, data, or statistics explicitly asked by the user."
    )
    final_answer: str = Field(
        description="The definitive, comprehensive, and complete final response combining all extracted information."
    )