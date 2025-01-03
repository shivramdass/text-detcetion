from ollama import chat
from pydantic import BaseModel
from typing import List, Optional, Literal

# Define the Object and ImageDescription classes
class Object(BaseModel):
    name: str
    confidence: float
    attributes: Optional[str] = None

class ImageDescription(BaseModel):
    detected_text: str  # Focus on detecting and extracting text
    objects: Optional[List[Object]] = None  # Optional, as objects may not be the focus
    scene: Optional[str] = None  # Scene information is secondary
    colors: Optional[List[str]] = None  # Optional, not relevant for handwritten text
    setting: Optional[Literal['Indoor', 'Outdoor', 'Unknown']] = None  # Optional

# Define the function to analyze image for handwritten text
def analyze_image_for_text(image_path: str) -> str:
    # Interaction with the model
    response = chat(
        model='llama3.2-vision',
        format=ImageDescription.model_json_schema(),  # Use updated schema
        messages=[{
            'role': 'user',
            'content': (
                "Analyze this image and detect any handwritten text. "
                "Provide the text exactly as seen and any relevant confidence scores."
            ),
            'images': [image_path],  # Provide the image path
        }],
        options={'temperature': 0},  # Set temperature to 0 for deterministic output
    )

    # Validate and parse the model response
    image_description = ImageDescription.model_validate_json(response.message.content)

    # Return detected text
    return image_description.detected_text
