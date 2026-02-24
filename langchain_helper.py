from typing import TypedDict, Sequence
import os
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
os.getenv('GEMINI_API_KEY')

CHROMA_PATH = "chroma"
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

db = Chroma(persist_directory=CHROMA_PATH,
            embedding_function=embeddings)

retriever = db.as_retriever(search_type="similarity")

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', google_api_key=GEMINI_API_KEY, temperature=0.2)

def contextualize_question():
    question_reformulation_prompt = """
    Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    question_reformulation_template = ChatPromptTemplate.from_messages(
        [
            ("system", question_reformulation_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, question_reformulation_template
    )

    return history_aware_retriever

def answer_question():
    answer_question_prompt = """
    Use the following pieces of retrieved context to answer the question. \
    Use three to seven sentences maximum and keep the answer concise, while still giving depth.\

    {context}"""

    answer_question_template = ChatPromptTemplate.from_messages(
        [
            ("system", answer_question_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    answer_question_chain = create_stuff_documents_chain(llm, answer_question_template)

    history_aware_retriever = contextualize_question()

    rag_chain = create_retrieval_chain(history_aware_retriever, answer_question_chain)

    return rag_chain

class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def call_model(state: State):
    rag_chain = answer_question()
    response = rag_chain.invoke(state)

    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

def execute_user_query(query_text):
    config = {"configurable": {"thread_id": "abc123"}}

    result = app.invoke({"input": query_text}, config=config, )

    return result["answer"]

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)