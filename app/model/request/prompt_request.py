from pydantic import BaseModel
from typing import List

class ProductDTO(BaseModel):
    name: str
    description: str
