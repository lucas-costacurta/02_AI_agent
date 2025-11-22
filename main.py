from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()  # Load environment variables from .env file

llm_gpt = ChatOpenAI(model_name="gpt-4", temperature=0) # Initialize OpenAI LLM
llm_anthropic = ChatAnthropic(model="claude-3-5-sonnet-20241022")  # Initialize Anthropic LLM

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
    llm=llm_anthropic,
    prompt=prompt,
    tools=[],  # Adicione ferramentas específicas aqui se necessário
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": "Impacto das mudanças climáticas na biodiversidade global"})
print("Resposta bruta do agente:", raw_response)
