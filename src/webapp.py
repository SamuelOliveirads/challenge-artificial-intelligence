import chainlit as cl
import httpx

API_URL = "http://localhost:8000/query"


@cl.on_chat_start
def start():
    cl.user_session.set("stage", "intro")


@cl.on_message
async def main(message: cl.Message) -> cl.Message:
    stage = cl.user_session.get("stage")
    question = message.content

    payload = {"question": question, "stage": stage}

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            response_message = result["message"]
            retrieved_docs = result.get("rag_content", None)
        except httpx.HTTPStatusError as e:
            response_message = f"Ocorreu um erro ao consultar a API: {str(e)}"
            retrieved_docs = None
        except httpx.RequestError as e:
            response_message = f"Erro na requisição: {str(e)}"
            retrieved_docs = None

    if stage == "intro":
        cl.user_session.set("stage", "main")
    elif stage == "main" and not retrieved_docs:
        cl.user_session.set("stage", "end")

    await cl.Message(content=response_message).send()
