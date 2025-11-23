from dotenv import load_dotenv
from pydantic import BaseModel
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    #tools=[search_tool],
)

class ResearchResponse(BaseModel): 
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]   

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente de pesquisa especializado em fornecer resumos detalhados."
    ),
    (
        "human",
        "Forneça um resumo detalhado sobre o seguinte tópico:\n\n"
        "{topic}\n\n"
        "{format_instructions}"
    )
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm_gemini | parser

query = input("O que deseja saber? ")
response = chain.invoke({"topic": query})

print(response)


