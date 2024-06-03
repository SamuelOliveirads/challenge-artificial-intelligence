import logging
import os
import uuid
from typing import Optional, Tuple

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.memory import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.llm.dinamic_state import (
    ContentRetrievalManager,
    ConversationCoordinator,
    StateController,
)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class StudyJourney:
    def __init__(self, retriever, llm_type="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=llm_type, temperature=0)
        self.session_id = str(uuid.uuid4())
        self.history = ChatMessageHistory()
        self.document_manager = ContentRetrievalManager(retriever)
        self.chatbot = ConversationCoordinator(self.document_manager)
        self.state_agent = StateController(self.chatbot)

        self.main_prompt_template = PromptTemplate(
            template="""
        Você é um assistente de aprendizado interativo da +A Educação. Seu objetivo é
        ajudar os usuários a se familiarizarem com o sistema e entenderem como você
        pode ajudá-los.

        Se o usuário disser "oi" ou perguntar "o que você pode fazer?", responda de
        maneira acolhedora explicando suas funcionalidades:
        1. Avaliar dificuldades de conhecimento.
        2. Perguntar sobre preferências de aprendizado.
        3. Fornecer sugestões de conteúdos adaptados.

        Aqui está a questão do usuário: {question}
        """,
            input_variables=["question"],
        )

        self.main_chain = LLMChain(
            prompt=self.main_prompt_template,
            llm=self.llm,
            output_parser=StrOutputParser(),
        )
        self.chain_with_history = RunnableWithMessageHistory(
            self.main_chain,
            get_session_history=lambda session_id: self.history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def add_to_history(self, sender: str, message: str):
        if sender == "user":
            self.history.add_user_message(message)
        else:
            self.history.add_ai_message(message)

    def run_interaction(self, question: str, document: str) -> Optional[str]:
        """Executes the interaction with the LLM, processing the given question and document details."""
        try:
            response = self.chain_with_history.invoke(
                {"question": question, "document": document},
                {"configurable": {"session_id": self.session_id}},
            )
        except AttributeError as e:
            logging.error(f"Error during LLM interaction: {str(e)}")
            response = "No response available."
            raise
        return response

    def update_prompt(self, next_prompt: PromptTemplate):
        self.main_prompt_template = next_prompt
        self.main_chain.prompt = self.main_prompt_template

    def get_answer(self, question: str) -> Tuple[str, Optional[str]]:
        """
        Get an answer from the LLM based on the stage of interaction.

        This function retrieves relevant documents and formats them, then uses
        different chains to get an answer based on the stage (intro, main, end).

        Parameters
        ----------
        question : str
            The question asked by the user.
        retriever : Chroma
            The retriever used to get relevant documents.
        stage : str, optional
            The stage of interaction, by default "main".

        Returns
        -------
        Tuple[str, Optional[str]]
            A tuple containing the response message and the formatted documents
            (if applicable).
        """
        self.add_to_history("user", question)
        retrieved_docs = self.document_manager.get_product_details(question)
        formatted_docs = self.document_manager.format_docs(retrieved_docs)
        next_prompt = self.state_agent.handle_input(self.history)
        self.update_prompt(next_prompt)
        response = self.run_interaction(question, formatted_docs)
        response_text = response.get("text", "Sem resposta disponível.")
        self.add_to_history("ai", response_text)

        return response
