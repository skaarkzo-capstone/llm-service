from pydantic import BaseModel

# For request structure
class PromptRequest(BaseModel):
    input_text: str