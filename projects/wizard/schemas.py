from pydantic import BaseModel, Field
from typing import List, Optional

class QueryResult(BaseModel):
    title: str = None
    url: str = None
    resume: str = None

class ReportState(BaseModel):
    user_input: str = None 
    final_response: str = None
    queries: List[str] = []