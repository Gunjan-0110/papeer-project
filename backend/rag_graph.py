import sqlite3
import operator
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from backend.models import RouterDecision, RelevancyDecision, ClaimVerificationResult
from backend.vector_store import search

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    query: str
    route: str
    retrieved_docs: list
    retrieval_attempts: int
    claim_verdict: str
    claim_source: str
    superseding_papers: list
    answer: str
    is_relevant: bool
    rewrite_count: int

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Define the routing prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert router. Decide whether the user query requires looking up the uploaded research paper/document ('vectorstore') or web search ('web_search')."),
    ("human", "{query}")
])

def route_query(state):
    """Route query to vectorstore or web search with dictionary state return."""
    print("---ROUTE QUERY---")
    
    try:
        decision = (prompt | llm.with_structured_output(RouterDecision)).invoke({"query": state["query"]})
        datasource = decision.datasource
    except Exception as e:
        print(f"Gemini API Server Error encountered: {e}. Defaulting to vectorstore.")
        datasource = "vectorstore"
        
    if datasource == "vectorstore":
        print("---ROUTE QUERY TO VECTORSTORE---")
        return {"route": "vectorstore"}
    elif datasource == "web_search":
        print("---ROUTE QUERY TO WEB SEARCH---")
        return {"route": "web_search"}
    else:
        return {"route": "vectorstore"}

def retrieve_docs(state: GraphState):
    docs = search(state["query"], state["session_id"], k=4)
    attempts = state.get("retrieval_attempts", 0) + 1
    return {"retrieved_docs": docs, "retrieval_attempts": attempts}


def check_relevancy(state: GraphState):
    docs = state.get("retrieved_docs", [])
    query = state["query"]
    context = "\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Decide if the following context is relevant to the user's query. It is relevant if it contains any information that helps answer the query."),
        ("human", "Query: {query}\n\nContext:\n{context}")
    ])
    decision = (prompt | llm.with_structured_output(RelevancyDecision)).invoke({"query": query, "context": context})
    return {"is_relevant": decision.is_relevant}


def rewrite_query(state: GraphState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's query to make it better for vector retrieval. The previous retrieval yielded irrelevant results."),
        ("human", "{query}")
    ])
    rewritten = (prompt | llm).invoke({"query": state["query"]}).content
    rewrites = state.get("rewrite_count", 0) + 1
    return {"query": rewritten, "rewrite_count": rewrites}


def verify_claim(state: GraphState):
    ddg = DuckDuckGoSearchRun()
    query = state["query"]
    
    web_results = ddg.invoke(query)
    arxiv_results = ddg.invoke(f"site:arxiv.org {query}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are verifying a claim. Based on the web and ArXiv search results, return a structured verdict determining if the claim is superseded, a summary explanation, and any newer superseding papers found."),
        ("human", f"Claim to verify: {query}\n\nWeb Results: {web_results}\n\nArXiv Results: {arxiv_results}")
    ])
    
    verification = (prompt | llm.with_structured_output(ClaimVerificationResult)).invoke({})
    
    answer = f"**Verdict:** {verification.verdict_summary}\n\n**Is Superseded:** {verification.is_superseded}\n\n"
    if verification.superseding_papers:
        answer += "**Superseding Papers:**\n"
        for p in verification.superseding_papers:
            answer += f"- [{p.title}]({p.url}): {p.summary}\n"
            
    return {
        "answer": answer, 
        "claim_verdict": verification.verdict_summary, 
        "superseding_papers": [p.model_dump() for p in verification.superseding_papers]
    }


def generate_answer(state: GraphState):
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    
    if docs and state.get("route") != "direct_answer":
        context = "\n\n".join(d.page_content for d in docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's question based strictly on the provided context."),
            ("human", f"Context:\n{context}\n\nQuestion: {query}")
        ])
        answer = (prompt | llm).invoke({"context": context, "query": query}).content
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's general knowledge question directly."),
            ("human", "{query}")
        ])
        answer = (prompt | llm).invoke({"query": query}).content
        
    return {"answer": answer}


def route_after_router(state: GraphState):
    return state["route"] if state["route"] in ["verify_claim", "direct_answer"] else "retrieve_docs"


def route_after_relevancy(state: GraphState):
    if state["is_relevant"] or state.get("rewrite_count", 0) >= 3:
        return "generate_answer"
    return "rewrite_query"


def build_graph(db_path: str = "checkpoints.db"):
    workflow = StateGraph(GraphState)
    
    workflow.add_node("route_query", route_query)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("check_relevancy", check_relevancy)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("verify_claim", verify_claim)
    workflow.add_node("generate_answer", generate_answer)
    
    workflow.set_entry_point("route_query")
    
    workflow.add_conditional_edges("route_query", route_after_router)
    workflow.add_edge("retrieve_docs", "check_relevancy")
    workflow.add_conditional_edges("check_relevancy", route_after_relevancy)
    workflow.add_edge("rewrite_query", "retrieve_docs")
    workflow.add_edge("verify_claim", END)
    workflow.add_edge("generate_answer", END)
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    
    return workflow.compile(checkpointer=memory)