import os
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 若后续动态切分可复用
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from src.settings import (
    CHROMA_PATH,
    COLLECTION_NAME,
    HF_MODEL_NAME,
    DEVICE,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
)

def build_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
    return store.as_retriever(search_kwargs={"k": 4})

def build_model():
    return ChatOpenAI(
        model="qwen3-next-80b-a3b-instruct",
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=0.2,
    )

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名农业土壤学专家助手。请基于提供的上下文回答问题，若上下文不足以回答，请说明原因并提出需要补充的信息。"),
        ("user", "问题：{question}\n\n参考资料：\n{context}\n\n请用中文给出结构化回答。"),
    ]
)

def format_docs(docs):
    return "\n\n".join(f"[来源页 {doc.metadata.get('page', 'N/A')}] {doc.page_content}" for doc in docs)

def build_rag_chain():
    retriever = build_retriever()
    model = build_model()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | model
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    rag_chain = build_rag_chain()
    question = "东北黑土区的有机质管理要点是什么？"
    answer = rag_chain.invoke(question)
    print(answer)