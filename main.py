from fastapi import FastAPI, HTTPException
import json
import os
import re
import requests
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

app = FastAPI(title="RAG & Agent-Based Chatbot",
    description="A chatbot utilizing Retrieval-Augmented Generation (RAG) and agent-based API integration for financial queries."
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG & Agent-Based Chatbot API. Use /ask to submit your questions."}


# LLM definitions
local_llm = "llama3.1:8b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0.1)
llm_json_mode = ChatOllama(model=local_llm, temperature=0.1, format="json")

llm_financial = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key='09772e4a1c0dcb0b284c2751fc7c4a7db638dfc5ac2a87037daa1eb73c1f6918',
    model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
)

# Load PDF documents from the "data" folder
DATA_PATH = "data"
loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
docs_list = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=350, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

# Create FAISS vectorstore from PDF chunks
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = FAISS.from_documents(doc_splits, embedding_model)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

router_instructions = """You are an expert at routing a user question to a vectorstore, a financial news agent, or web search.
The vectorstore contains documents related to financial reports to saudi stock companies (Aramco, Elm, ...).
The financial news agent uses an API to fetch the latest stock news.
For questions about financial markets, stock analysis, or that mention a stock ticker (e.g. AAPL, TSLA), use the financial news agent.
For current events or other topics, use web-search.
Return JSON with single key, datasource, that is 'websearch', 'vectorstore', or 'financialnews' depending on the question."""
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with a single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""
rag_prompt = """You are an assistant for question-answering tasks.

Here is the context to use to answer the question:

{context}

Think carefully about the above context.

Now, review the user question:

{question}

Provide an answer to this question using only the above context.

Use three sentences maximum and keep the answer concise.

Answer:"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

hallucination_grader_instructions = """
You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:
A score of yes means that the student's answer meets all of the criteria.
A score of no means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset.
"""
hallucination_grader_prompt = """FACTS: \n\n {documents}\n\nSTUDENT ANSWER: {generation}.
Return JSON with two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS, and explanation that contains an explanation of the score (non-empty)."""
answer_grader_instructions = """You are a teacher grading a quiz.
You will be given a QUESTION and a STUDENT ANSWER.
Here is the grade criteria to follow:
(1) The STUDENT ANSWER helps to answer the QUESTION.
Score:
A score of yes means that the student's answer meets all of the criteria.
A score of no means that the student's answer does not meet the criteria.
The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset."""
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}.
Return JSON with two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria, and explanation that contains an explanation of the score."""

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = "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"

_set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

web_search_tool = TavilySearchResults(k=3)

class FinancialNewsTool:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_latest_news(self, stock_symbol: str) -> str:
        today = datetime.today()
        week_ago = today - timedelta(days=7)
        today_str = today.strftime('%Y-%m-%d')
        week_ago_str = week_ago.strftime('%Y-%m-%d')
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={stock_symbol}&from={week_ago_str}&to={today_str}&token={self.api_key}"
        )
        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json()
            sorted_news = sorted(news_data, key=lambda x: x['datetime'], reverse=True)
            headlines = [
                f"{news['headline']} (Date: {datetime.fromtimestamp(news['datetime']).strftime('%Y-%m-%d')})"
                for news in sorted_news[:5]
            ]
            return "\n".join(headlines)
        else:
            return f"Error fetching news: {response.status_code}"

FINNHUB_API_KEY = "cupi05hr01qp7lft1gigcupi05hr01qp7lft1gj0"
news_tool_instance = FinancialNewsTool(FINNHUB_API_KEY)

def financial_news_tool_func(stock_symbol: str) -> str:
    return news_tool_instance.get_latest_news(stock_symbol)



class GraphState(TypedDict):
    question: str
    generation: any
    web_search: str
    max_retries: int
    answers: int
    loop_step: Annotated[int, operator.add]
    documents: List[Document]

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    print(question)
    documents = retriever.invoke(question)
    return {"documents": documents}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question=question)
        result = llm_json_mode.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)["binary_score"]
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}

def answer_financial_news(state):
    print("---FINANCIAL NEWS NODE---")
    loop_step = state.get("loop_step", 0)
    tickers = re.findall(r'\b[A-Z]{1,5}\b', state["question"])
    ticker = tickers[0] if tickers else ""
    if not ticker:
        return {"generation": llm.invoke('The user did not provide a stock ticker, politely ask for the stock ticker.'), "documents": [], "loop_step": loop_step + 1}
    news = financial_news_tool_func(ticker)
    documents = [Document(page_content=news)]
    prompt = PromptTemplate.from_template("""You are a financial analyst. Based on the following recent financial news and the question, provide a detailed answer.

Ticker: {ticker}

Financial News:
{news}

Question: {input}

Your detailed answer:
  """)
    chain = prompt | llm_financial
    response = chain.invoke({"ticker": ticker, "news": news, "input": state["question"]})
    return {"generation": response, "documents": documents, "loop_step": loop_step + 1}

def route_question(state):
    print("---ROUTE QUESTION---")
    route_question = llm_json_mode.invoke([SystemMessage(content=router_instructions)] + [HumanMessage(content=state["question"])])
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "retrieve"
    elif source == "financialnews":
        print("---ROUTE QUESTION TO FINANCIAL NEWS---")
        return "financial_news"

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search_flag = state["web_search"]
    filtered_documents = state["documents"]
    if web_search_flag == "Yes":
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(documents=format_docs(documents), generation=generation.content)
    result = llm_json_mode.invoke([SystemMessage(content=hallucination_grader_instructions)] + [HumanMessage(content=hallucination_grader_prompt_formatted)])
    grade = json.loads(result.content)["binary_score"]
    if grade == "yes":
        print("---DECISION: GENERATION GROUNDED---")
        answer_grader_prompt_formatted = answer_grader_prompt.format(question=question, generation=generation.content)
        result = llm_json_mode.invoke([SystemMessage(content=answer_grader_instructions)] + [HumanMessage(content=answer_grader_prompt_formatted)])
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION NOT GROUNDED, RETRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"

workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("financial_news", answer_financial_news)
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "retrieve": "retrieve",
        "financial_news": "financial_news",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)
workflow.add_conditional_edges(
    "financial_news",
    grade_generation_v_documents_and_question,
    {
        "not supported": "websearch",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)
graph = workflow.compile()

class AskRequest(BaseModel):
    question: str
    max_retries: int = 3


@app.post("/ask")
def ask_question(request: AskRequest):
    print("-"*100)
    print(request)
    print("-"*100)
    try:
        input_data = request.model_dump()
        print(input_data)
        result = graph.invoke(input_data)
        generation = result.get("generation")
        # generation is expected to have a 'content' attribute
        answer_text = generation.content if hasattr(generation, "content") else str(generation)
        return {"answer": answer_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

