import chainlit as cl

from create_rag_db import update_chroma_db
from llm_model import create_llm, get_answer


@cl.on_chat_start
def start():
    retriever = update_chroma_db()
    retrieval_chain = create_llm()

    cl.user_session.set("retriever", retriever)
    cl.user_session.set("agent", retrieval_chain)


@cl.on_message
async def main(message):
    retriever = cl.user_session.get("retriever")
    agent = cl.user_session.get("agent")

    question = message.content
    response, retrieved_docs = get_answer(question, retriever, agent)

    await cl.Message(content=response).send()
    await cl.Message(content=f"Documentos Recuperados:\n{retrieved_docs}").send()
