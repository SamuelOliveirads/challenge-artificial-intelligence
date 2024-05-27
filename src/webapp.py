import chainlit as cl

from create_rag_db import update_chroma_db
from llm_model import StudyJourney


@cl.on_chat_start
def start():
    retriever = update_chroma_db()
    study_journey = StudyJourney()

    cl.user_session.set("retriever", retriever)
    cl.user_session.set("agent", study_journey)
    cl.user_session.set("stage", "intro")


@cl.on_message
async def main(message: cl.Message) -> cl.Message:
    retriever = cl.user_session.get("retriever")
    agent = cl.user_session.get("agent")
    stage = cl.user_session.get("stage")

    question = message.content
    response, retrieved_docs = agent.get_answer(question, retriever, stage)

    if stage == "intro":
        cl.user_session.set("stage", "main")
    elif stage == "main" and not retrieved_docs:
        cl.user_session.set("stage", "end")

    await cl.Message(content=response).send()
