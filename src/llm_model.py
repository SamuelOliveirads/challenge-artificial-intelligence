import os

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

os.environ[
    "OPENAI_API_KEY"
] = "sk-proj-"


def create_llm():
    llm_type = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_type, temperature=0)

    prompt_template = PromptTemplate(
        template="""
    Você é um assistente da empresa +A Educação onde vai receber
    usuários com dúvidas sobre o conteúdo de estudo. Ajude os usuários
    indicando conteúdos relevantes de acordo com o nível de dificuldade
    de cada um. Use exatamente o que estiver em: \n\n {document} \n\n
    Caso não tenha a informação, informe que não possui informação sobre
    o tema e sugira uma nova pergunta ao usuário.
    Aqui está a questão do usuário: {question} \n

    Responda de maneira clara e direta com base nas informações fornecidas.
    """,
        input_variables=["question", "document"],
    )

    retrieval_chain = LLMChain(
        prompt=prompt_template, llm=llm, output_parser=StrOutputParser()
    )

    return retrieval_chain


def format_docs(docs):
    formatted_docs = ""
    for i, doc in enumerate(docs):
        formatted_docs += f"Documento {i+1}:\n{doc.page_content}\n\n"
    return formatted_docs


def get_answer(question, retriever, retrieval_chain):
    # Recuperar os documentos relevantes
    retrieved_docs = retriever.get_relevant_documents(question)
    formatted_docs = format_docs(retrieved_docs)

    # Obter a resposta do modelo
    response = retrieval_chain.run(question=question, document=formatted_docs)
    return response, formatted_docs
