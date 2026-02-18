from pydantic import BaseModel, Field, ValidationError
from typing import List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
class ReviewAnalysis(BaseModel):
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    rating: int = Field(description="Rating from 1 to 5")
    key_features: List[str] = Field(description="Important product features mentioned")
    improvement_suggestions: List[str] = Field(description="Suggestions for improvement")

parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
prompt = PromptTemplate(
    template="""
You are an AI product review analyzer.

Return output strictly in JSON format.

{format_instructions}

Review:
{review_text}
""",
    input_variables=["review_text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

llm = OllamaLLM(model="llama3", temperature=0)

chain = prompt | llm

review = """
The smartphone has excellent battery life and a stunning display.
However, the camera quality is average and it feels overpriced.
"""
try:
    raw_output = chain.invoke({"review_text": review})
    result = parser.parse(raw_output)

    print("\nStructured Output:\n")
    print(result)

except ValidationError as ve:
    print("Validation Error:")
    print(ve)

except Exception as e:
    print("Other Error:")
    print(e)