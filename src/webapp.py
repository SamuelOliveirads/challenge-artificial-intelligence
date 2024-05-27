import chainlit as cl

from create_rag_db import update_chroma_db
from llm_model import StudyJourney


@cl.on_chat_start
def start():
    retriever = update_chroma_db()
    study_journey = StudyJourney()

    cl.user_session.set("retriever", retriever)
    cl.user_session.set("agent", study_journey)


@cl.on_message
async def main(message):
    retriever = cl.user_session.get("retriever")
    agent = cl.user_session.get("agent")

    question = message.content
    response, retrieved_docs = agent.get_answer(question, retriever)

    await cl.Message(content=response).send()
    await cl.Message(content=("Documentos Recuperados:" f"\n{retrieved_docs}")).send()
