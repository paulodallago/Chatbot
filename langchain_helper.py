import os

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Sequence
from typing_extensions import Annotated

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
os.getenv('GEMINI_API_KEY')

CHROMA_PATH    = "chroma"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_MODEL   = 'gemini-2.5-flash-lite'
GEMINI_MODEL   = 'gemini-2.5-flash'


gpt4all_embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={'allow_download': 'True'}
)

db = Chroma(persist_directory=CHROMA_PATH,
            embedding_function=gpt4all_embeddings)

retriever = db.as_retriever(search_type="similarity")

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0.2)

def contextualize_question():
    question_reformulation_prompt = """
    Dado o histórico do chat e a última pergunta, que pode referênciar algo do passado \ 
    formule uma única pergunta. NÃO RESPONDA a pergunta, apenas reformule \
    se necessário, ou retorne-a como estava. \
    """

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
    Você foi treinado com documentos do curso de Ciência da Computação do IFSUL Passo Fundo. \
    Você DEVE responder de acordo com esses documentos. Cite fontes. \
    Se não tiver a resposta nos documentos, você DEVE deixar claro que está usando conhecimento externo. \
    Você será um assistente para alunos jovens. Seja educado e alegre. \

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

    # if response

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

    print(">", query_text)
    print(">>", result["answer"])

    return result["answer"]

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)