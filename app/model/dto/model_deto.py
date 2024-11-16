from pydantic import BaseModel

# For request structure
class TextRequest(BaseModel):
    input_text: str