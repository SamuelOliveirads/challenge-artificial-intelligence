import json
import os
from pathlib import Path

import assemblyai as aai
import nest_asyncio
import pytesseract
from langchain.docstore.document import Document as LangchainDocument
from langchain.document_loaders import (
    AssemblyAIAudioTranscriptLoader,
    JSONLoader,
    TextLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from llama_parse import LlamaParse
from moviepy.editor import VideoFileClip
from PIL import Image

from utils import load_from_file, save_to_file

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""

os.environ[
    "OPENAI_API_KEY"
] = "sk-proj-"
os.environ[
    "LLAMA_CLOUD_API_KEY"
] = "llx-"
aai.settings.api_key = ""

# # 1.0 Load Data
def load_data():
    # ## 1.1 Load PDF Files
    llama_documents_file = "../data/02_intermediate/llama_documents.pkl"
    nest_asyncio.apply()

    # Configuração do LlamaParse
    parser = LlamaParse(
        result_type="text",  # Usar o método de texto
        num_workers=4,
        verbose=True,
        language="pt",
    )
    # Carregar e processar o documento PDF
    llama_documents = parser.load_data("../data/01_raw/Capítulo do Livro.pdf")

    save_to_file(llama_documents, llama_documents_file)
    print("Dados processados do PDF salvos com sucesso.")

    structured_documents = [
        LangchainDocument(page_content=doc.text, metadata=doc.metadata)
        for doc in llama_documents
    ]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    # Dividir os documentos em chunks menores
    docs = text_splitter.split_documents(structured_documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"

    # ## 1.2 Load Text files
    # Carregar arquivo de texto
    text_loader = TextLoader("../data/01_raw/Apresentação.txt")
    text_documents = text_loader.load()

    # Processar e inserir no RAG
    for doc in text_documents:
        doc.metadata["source"] = "Apresentação.txt"
    docs.extend(text_documents)

    # ## 1.3 Load JSON files
    json_path = "../data/01_raw/Exercícios.json"
    data = json.loads(Path(json_path).read_text())

    def process_json_data(data):
        processed_data = []
        metadata = {
            "_id": data["_id"]["$oid"],
            "external_id": data["external_id"],
            "name": data["name"],
            "external_topicId": data["external_topicId"],
            "title": data["title"],
            "type": data["type"],
            "language": data["language"],
            "author_name": data["author"]["name"],
            "author_alias": data["author"]["alias"],
            "tags": data["tags"],
            "banner_url": data["banner"]["url"],
            "created_at": data["created_at"]["$date"],
            "modified_at": data["modifed_at"]["$date"],
            "version": data["version"],
            "status": data["status"],
            "resource": data["resource"],
            "category": data["category"],
            "icon": data["icon"],
            "isReviewed": data["isReviewed"],
        }

        for question in data.get("content", []):
            question_metadata = metadata.copy()
            question_metadata.update(
                {
                    "question_id": question["_id"]["$oid"],
                    "question_title": question.get("title"),
                }
            )

            question_content = question.get("content", {})
            question_text = question_content.get("html", "")

            options = question_content.get("options", [])
            option_texts = []
            feedbacks = []
            correct_answers = []

            for option in options:
                option_text = option["content"].get("html", "")
                feedback = option["feedback"].get("html", "")
                correct = option.get("correct", False)

                option_texts.append(option_text)
                feedbacks.append(feedback)
                correct_answers.append(correct)

            combined_content = (
                f"Pergunta: {question_text}\nOpções:\n"
                + "\n".join(f"- {opt}" for opt in option_texts)
                + "\nFeedbacks:\n"
                + "\n".join(f"- {feed}" for feed in feedbacks)
                + "\nCorretas:\n"
                + "\n".join(f"- {correct}" for correct in correct_answers)
            )

            new_doc = LangchainDocument(
                page_content=combined_content, metadata=question_metadata
            )
            processed_data.append(new_doc)

        return processed_data

    # Processar os dados JSON
    processed_documents = process_json_data(data)

    with open("../data/02_intermediate/processed_questions.json", "w") as f:
        json.dump([doc.dict() for doc in processed_documents], f)

    # Carregar os documentos processados usando JSONLoader
    json_loader = JSONLoader(
        file_path="../data/02_intermediate/processed_questions.json",
        jq_schema=".",
        text_content=False,
    )
    json_documents = json_loader.load()

    # Processar e inserir no RAG
    for doc in json_documents:
        doc.metadata["source"] = "Exercícios.json"
    docs.extend(json_documents)

    # ## 1.4 Load Image files
    def extract_text_from_image(image_path, lang="por"):
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=lang)
        return text

    # Carregar e processar a imagem
    image_path = "../data/01_raw/Infografico-1.jpg"
    extracted_text = extract_text_from_image(image_path)

    # Criar documento com o texto extraído
    image_document = LangchainDocument(
        page_content=extracted_text, metadata={"source": "Infografico-1.jpg"}
    )

    # Processar e inserir no RAG
    docs.append(image_document)

    # ## 1.5 Load Video files
    video_file = "../data/01_raw/Dica_do_professor.mp4"
    audio_file = "../data/02_intermediate/Dica_do_professor.mp3"

    video = VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)

    # Carregar e transcrever o áudio
    config = aai.TranscriptionConfig(language_code="pt")
    audio_loader = AssemblyAIAudioTranscriptLoader(file_path=audio_file, config=config)
    audio_documents = audio_loader.load()

    # Processar e inserir no RAG
    for doc in audio_documents:
        doc.metadata["source"] = "Dica do professor.mp4"
    docs.extend(audio_documents)

    # ## 1.6 Cleaning metadata
    def find_and_correct_invalid_metadata(documents):
        problematic_docs = []
        for doc in documents:
            keys_to_remove = []
            for key, value in doc.metadata.items():
                if value is None or (isinstance(value, list) and len(value) == 0):
                    problematic_docs.append((key, value, doc))
                    # Corrigir metadados
                    if value is None:
                        doc.metadata[key] = "unknown"
                    elif isinstance(value, list) and len(value) == 0:
                        doc.metadata[key] = "empty_list"
                elif isinstance(value, list) and all(
                    isinstance(item, dict) for item in value
                ):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                problematic_docs.append((key, doc.metadata[key], doc))
                del doc.metadata[key]

        return problematic_docs

    # Localizar e corrigir documentos com metadados inválidos
    problematic_docs = find_and_correct_invalid_metadata(docs)

    # Exibir resultados
    if problematic_docs:
        print("Documentos com metadados inválidos encontrados e corrigidos:")
        for key, value, doc in problematic_docs:
            print(f"Documento com {key} = {value} foi corrigido")
    else:
        print("Nenhum documento com metadados inválidos encontrado.")

    compress_documents_files = "../data/03_primary/compress_documents.pkl"

    save_to_file(docs, compress_documents_files)

    docs = load_from_file(compress_documents_files)

    chroma_db_dir = "../data/03_primary/chroma_db"
    docsearch = Chroma.from_documents(
        docs,
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory=chroma_db_dir,
    )
    print("Chroma DB salvo com sucesso.")
    return docs, docsearch
