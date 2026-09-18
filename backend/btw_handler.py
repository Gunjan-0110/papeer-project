from typing import Generator

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

from backend.models import BtwRouteDecision

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def handle_btw(query: str) -> Generator[str, None, None]:
    """Off-topic side channel — never touches the vector store or checkpointer."""
    route_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Decide if answering this question requires a real-time web search (recent events, "
         "current prices, breaking news) or if your general knowledge is sufficient."),
        ("human", "{query}"),
    ])
    
    decision = (route_prompt | llm.with_structured_output(BtwRouteDecision)).invoke({"query": query})

    if decision.needs_web_search:
        ddg = DuckDuckGoSearchRun()
        context = ddg.invoke(query)
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using the web search results below. Be concise.\n\n"
             f"Results:\n{context}"),
            ("human", "{query}"),
        ])
    else:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question concisely from your general knowledge."),
            ("human", "{query}"),
        ])

    for chunk in (answer_prompt | llm).stream({"query": query}):
        if chunk.content:
            yield chunk.content