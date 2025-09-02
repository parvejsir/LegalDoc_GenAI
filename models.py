from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class LegalDocSummary(BaseModel):
    category: str = Field(..., description="Category/type of the legal document (e.g., summons, contract, agreement, notice).")
    description: str = Field(..., description="2-line description/overview of the document.")
    important_timeline: List[str] = Field(..., description="Key dates, deadlines, or timeline events mentioned in the document.")
    main_takeaway: List[str] = Field(..., description="Primary conclusions or essence of the document in bullet points.")
    risk_factors: List[str] = Field(..., description="Possible risks, consequences, or liabilities arising from the document.")
    next_steps: List[str] = Field(..., description="Recommended actions or solutions to mitigate the risks or comply with the document.")
    urgency_percentage: int = Field(..., ge=0, le=100, description="Numerical urgency score between 0-100.")
    urgency_level: str = Field(..., description="Categorical urgency level (e.g., High, Medium, Low).")

class SummaryRequest(BaseModel):
    document_content: str
    language: str

class SummaryResponse(BaseModel):
    summary: LegalDocSummary
    is_summarized: bool

class LastDateResponse(BaseModel):
    last_date: Optional[date] = Field(..., description="The last date to take action mentioned in the document.")