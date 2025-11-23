from dotenv import load_dotenv
from pydantic import BaseModel
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()  # Load environment variables from .env file

gemini_key = os.getenv("GEMINI_API_KEY")

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)  # Initialize Gemini LLM

class ResearchResponse(BaseModel): 
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]   

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente de pesquisa especializado em fornecer resumos detalhados sobre tópicos variados."),
        (
            "user",
            "Forneça um resumo detalhado sobre o seguinte tópico, incluindo fontes confiáveis e ferramentas utilizadas na pesquisa: {topic}\n\n"
            "Formato da resposta:\n"
            "- topic: Tópico pesquisado\n"
            "- summary: Resumo detalhado\n"
            "- sources: Lista de fontes confiáveis\n"
            "- tools_used: Lista de ferramentas utilizadas na pesquisa\n\n"
            "{format_instructions}",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ],
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm_gemini,
    prompt=prompt,
    tools=[],  # Adicione ferramentas específicas aqui se necessário
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": input("O que deseja saber? ")})
print(raw_response)
