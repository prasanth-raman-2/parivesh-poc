
from pydantic import BaseModel, Field


class ECModel(BaseModel):
    project_name: str = Field(None, description="Name of the project")
    description: str = Field(None, description="Description of the project")
    category: str = Field(None, description="Category of the project: A, B1, or B2")
    