from typing import List

from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import UserProxyAgent
from langchain.docstore.document import Document as LangchainDocument
from langchain.memory import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma


class ContentRetrievalManager:
    def __init__(self, retriever: Chroma):
        self.retriever = retriever

    def format_docs(self, docs: List[LangchainDocument]) -> str:
        """
        Format documents into a structured string.

        This function formats a list of documents into a structured string,
        including the source and type of each document (e.g., Vídeo, PDF,
        Texto, Exercício, Imagem).

        Parameters
        ----------
        docs : List[LangchainDocument]
            The list of documents to be formatted.

        Returns
        -------
        str
            A formatted string representing the documents.
        """
        formatted_docs = ""
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Desconhecido")
            if source.endswith(".mp4"):
                format_type = "Vídeo"
            elif source.endswith(".pdf"):
                format_type = "PDF"
            elif source.endswith(".txt"):
                format_type = "Texto"
            elif source.endswith(".json"):
                format_type = "Exercício"
            elif source.endswith(".jpg") or source.endswith(".png"):
                format_type = "Imagem"
            else:
                format_type = "Desconhecido"

            formatted_docs += (
                f"Documento {i+1} ({format_type}):" f"\n{doc.page_content}\n\n"
            )
        return formatted_docs

    def get_product_details(self, query: str) -> str:
        """
        Retrieve product details from RAG based on the query.

        Parameters
        ----------
        query : str
            The user's query or product name to search for in the RAG.

        Returns
        -------
        str
            String with product details.
        """
        retrieved_docs = self.retriever.get_relevant_documents(query)
        return retrieved_docs


class ConversationCoordinator:
    """
    A chatbot class designed to manage interactions within an educational environment.
    This chatbot assists users by providing personalized content recommendations based on their queries
    and guiding them through the various stages of the learning process.

    Attributes:
        state (str): Represents the current state of the chatbot in the conversation flow.
        document_manager (ContentRetrievalManager): Manages retrieval and formatting of product
                                           details from a document database.
        prompts (dict): A dictionary mapping conversation states to their respective
                        prompt templates, which are used to generate responses based on
                        user inputs and document data.

    Methods:
        __init__: Initializes the chatbot with a document retriever.
    """

    def __init__(self, retriever: Chroma):
        self.state = "intro"
        self.document_manager = ContentRetrievalManager(retriever)
        self.prompts = {
            "intro": PromptTemplate(
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
            ),
            "main": PromptTemplate(
                template="""
                    Você é um assistente de aprendizado interativo da +A Educação. Seu objetivo é
                    ajudar os usuários a identificar suas dificuldades e lacunas de conhecimento em
                    um tema específico e fornecer conteúdos adaptados ao seu nível de conhecimento
                    e formato de preferência.

                    Durante o diálogo, você deve:
                    1. Avaliar e entender as áreas onde o conhecimento do usuário pode ser insuficiente.
                    2. Perguntar sobre as preferências de aprendizado do usuário (texto, vídeo, áudio).
                    3. Fornecer respostas e sugestões de conteúdos adaptados às preferências e
                    necessidades do usuário, usando exatamente o que estiver em: \n\n {document} \n\n.

                    Aqui está a questão do usuário: {question}

                    Responda de maneira clara e direta com base nas informações fornecidas e adapte suas
                    respostas conforme as preferências de aprendizado do usuário.
                    Caso não tenha a informação, informe que não possui informação sobre o tema e
                    sugira uma nova pergunta ao usuário.
                    """,
                input_variables=["question", "document"],
            ),
            "end": PromptTemplate(
                template="""
        Você é um assistente de aprendizado interativo da +A Educação. Seu objetivo é
        garantir que o usuário não tenha mais dúvidas pendentes e se despedir de maneira cordial.

        Pergunte ao usuário se ele tem mais alguma dúvida. Se não, agradeça pela interação
        e deseje ótimos estudos, caso o usuário agradeça também o agradeça.

        Aqui está a questão do usuário: {question}
        """,
                input_variables=["question"],
            ),
        }


class StateController:
    """
    Manages conversation state transitions within a marketplace environment, utilizing
    a state machine approach with an LLM to determine and update states based on user interaction.

    Attributes:
        chatbot (ConversationCoordinator): The chatbot instance managing the conversation.
        llm (RetrieveAssistantAgent): LLM agent used to determine conversation state.
        visited_states (List[str]): A list of states the conversation has already visited.
        user_proxy (UserProxyAgent): Proxy agent that manages communication with the LLM.
    """

    def __init__(self, chatbot: ConversationCoordinator):
        self.chatbot = chatbot
        self.llm = RetrieveAssistantAgent(
            name="MarketplaceStateAgent",
            system_message="Determine the current state of the conversation based on the history provided.",
            llm_config={
                "timeout": 600,
                "cache_seed": 42,
                "config_list": [{"model": "gpt-3.5-turbo", "temperature": 0}],
            },
        )
        self.visited_states = []
        self.user_proxy = UserProxyAgent(
            name="state_agent",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            is_termination_msg=lambda x: x.get("content", "")
            .rstrip()
            .endswith("TERMINATE")
            or x.get("content", "").rstrip().endswith("TERMINATE."),
            code_execution_config={
                "use_docker": False,
            },
        )

    def determine_state(self, history: ChatMessageHistory) -> str:
        """
        Determines the current conversation state based on the provided history.

        Parameters:
            history (ChatMessageHistory): The historical record of the conversation.

        Returns:
            str: The predicted current state of the conversation.
        """
        prompt = self.generate_prompt(history)
        state_prediction = self.user_proxy.initiate_chat(self.llm, message=prompt)
        return state_prediction

    def generate_prompt(self, history: ChatMessageHistory) -> str:
        visited_states = ", ".join(self.visited_states)
        prompt = f"""
        Given the conversation history:
        '{history}'

        Determine the current stage of the marketplace process. The stages are in order:
        intro > main > end.

        The user has already visited these stages: '{visited_states}'

        What is the current stage? Reply only the specified stage name.
        """
        return prompt

    def update_chatbot_state(self, history: ChatMessageHistory) -> str:
        """
        Updates the chatbot's state based on the conversation history.

        Parameters:
            history (ChatMessageHistory): The historical record of the conversation.

        Returns:
            str: The chatbot's prompt for the newly updated state.
        """
        predicted_state = self.determine_state(history)
        llm_response = predicted_state.summary
        if llm_response not in self.chatbot.state:
            self.visited_states.append(llm_response)
            self.chatbot.state = llm_response
        return self.chatbot.prompts[llm_response]

    def handle_input(self, history: ChatMessageHistory) -> str:
        return self.update_chatbot_state(history)
