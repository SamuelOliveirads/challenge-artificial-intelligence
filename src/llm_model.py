import os

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = "sk-proj-"


class StudyJourney:
    def __init__(self, llm_type="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=llm_type, temperature=0)
        self.prompt_template = PromptTemplate(
            template="""
        Você é um assistente de aprendizado interativo da +A Educação.
        Seu objetivo é ajudar os usuários a identificar suas dificuldades
        e lacunas de conhecimento em um tema específico e fornecer
        conteúdos adaptados ao seu nível de conhecimento e formato de
        preferência.

        Durante o diálogo, você deve:
        1. Avaliar e entender as áreas onde o conhecimento do usuário
        pode ser insuficiente.
        2. Perguntar sobre as preferências de aprendizado do usuário
        (texto, vídeo, áudio).
        3. Fornecer respostas e sugestões de conteúdos adaptados às
        preferências e necessidades do usuário, usando exatamente o que
        estiver em: \n\n {document} \n\n.

        Aqui está a questão do usuário: {question} \n

        Responda de maneira clara e direta com base nas informações
        fornecidas e adapte suas respostas conforme as preferências de
        aprendizado do usuário.
        Caso não tenha a informação, informe que não possui informação
        sobre o tema e sugira uma nova pergunta ao usuário.
        """,
            input_variables=["question", "document"],
        )

        self.retrieval_chain = LLMChain(
            prompt=self.prompt_template, llm=self.llm, output_parser=StrOutputParser()
        )

    def format_docs(self, docs):
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

    def get_answer(self, question, retriever):
        # Recuperar os documentos relevantes
        retrieved_docs = retriever.get_relevant_documents(question)
        formatted_docs = self.format_docs(retrieved_docs)

        # Obter a resposta do modelo
        response = self.retrieval_chain.run(question=question, document=formatted_docs)
        return response, formatted_docs
