# summarizer.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.pydantic import PydanticOutputParser
from models import LegalDocSummary

# Move model initialization into a function that takes the API key
def get_model(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        max_output_tokens=8192,
        google_api_key=api_key
    )

def generate_document_summary(document_content: str, language: str, google_api_key: str) -> LegalDocSummary:
    """
    Generates a structured summary of a document using Gemini 2.0 Flash.
    """
    model = get_model(google_api_key)
    
    # Create the structured output parser
    parser = PydanticOutputParser(pydantic_object=LegalDocSummary)

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert legal assistant. Summarize the following legal document in a structured format."),
        ("human", "Summarize the following document in {language}. "
         "The document content is: \n\n{document_content}\n\n"
         "{format_instructions}"
        )
    ]).partial(format_instructions=parser.get_format_instructions())

    # Create the chain and invoke it
    chain = prompt | model | parser
    
    try:
        summary = chain.invoke({
            "language": language,
            "document_content": document_content
        })
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return LegalDocSummary(
            category="Error",
            description="Could not generate a summary due to an error.",
            important_timeline=[],
            main_takeaway=[f"An error occurred: {str(e)}"],
            risk_factors=[],
            next_steps=[],
            urgency_percentage=0,
            urgency_level="Low"
        )