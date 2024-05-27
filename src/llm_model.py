import os
from typing import List, Optional, Tuple

from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document as LangchainDocument
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = "sk-proj-"


class StudyJourney:
    def __init__(self, llm_type="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=llm_type, temperature=0)

        self.intro_prompt_template = PromptTemplate(
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

        self.main_prompt_template = PromptTemplate(
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
        )

        self.end_prompt_template = PromptTemplate(
            template="""
        Você é um assistente de aprendizado interativo da +A Educação. Seu objetivo é
        garantir que o usuário não tenha mais dúvidas pendentes e se despedir de maneira cordial.

        Pergunte ao usuário se ele tem mais alguma dúvida. Se não, agradeça pela interação
        e deseje ótimos estudos, caso o usuário agradeça também o agradeça.

        Aqui está a questão do usuário: {question}
        """,
            input_variables=["question"],
        )

        self.intro_chain = LLMChain(
            prompt=self.intro_prompt_template,
            llm=self.llm,
            output_parser=StrOutputParser(),
        )
        self.main_chain = LLMChain(
            prompt=self.main_prompt_template,
            llm=self.llm,
            output_parser=StrOutputParser(),
        )
        self.end_chain = LLMChain(
            prompt=self.end_prompt_template,
            llm=self.llm,
            output_parser=StrOutputParser(),
        )

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

    def get_answer(
        self, question: str, retriever: Chroma, stage: str = "main"
    ) -> Tuple[str, Optional[str]]:
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
        if stage == "intro":
            message = self.intro_chain.run(question=question)
            rag_content = None
        elif stage == "main":
            retrieved_docs = retriever.get_relevant_documents(question)
            formatted_docs = self.format_docs(retrieved_docs)
            message = self.main_chain.run(question=question, document=formatted_docs)
            rag_content = formatted_docs
        elif stage == "end":
            message = self.end_chain.run(question=question)
            rag_content = None

        return message, rag_content
