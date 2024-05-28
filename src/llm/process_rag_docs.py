import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

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

os.environ["OPENAI_API_KEY"] = "sk-proj-"
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-"
aai.settings.api_key = ""

logs = logging.basicConfig(level=logging.INFO)


def process_json_data(data: dict) -> List[LangchainDocument]:
    """Process JSON data and create structured documents.

    Parameters
    ----------
    data : dict
        The JSON data to be processed.

    Returns
    -------
    List[LangchainDocument]
        A list of structured documents.
    """
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


def extract_text_from_image(image_path: str, lang: str = "por") -> str:
    """Extract text from an image using OCR.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    lang : str, optional
        The language for OCR processing, by default "por".

    Returns
    -------
    str
        The extracted text.
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=lang)
    return text


def find_and_correct_invalid_metadata(
    documents: List[LangchainDocument],
) -> List[Tuple[str, any, LangchainDocument]]:
    """Find and correct invalid metadata in documents.

    Parameters
    ----------
    documents : List[LangchainDocument]
        The list of documents to be checked and corrected.

    Returns
    -------
    List[Tuple[str, any, LangchainDocument]]
        A list of tuples containing the key, value, and document
        with problematic metadata.
    """
    problematic_docs = []
    for doc in documents:
        keys_to_remove = []
        for key, value in doc.metadata.items():
            if value is None or (isinstance(value, list) and len(value) == 0):
                problematic_docs.append((key, value, doc))
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


def load_pdf() -> List[LangchainDocument]:
    """Load and process PDF files.

    Returns
    -------
    List[LangchainDocument]
        A list of structured documents from PDF files.
    """
    llama_documents_file = "../data/02_intermediate/llama_documents.pkl"
    nest_asyncio.apply()

    parser = LlamaParse(
        result_type="text",
        num_workers=4,
        verbose=True,
        language="pt",
    )
    llama_documents = parser.load_data("../data/01_raw/Capítulo do Livro.pdf")

    save_to_file(llama_documents, llama_documents_file)
    logs.info("Dados processados do PDF salvos com sucesso.")

    structured_documents = [
        LangchainDocument(page_content=doc.text, metadata=doc.metadata)
        for doc in llama_documents
    ]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    docs = text_splitter.split_documents(structured_documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs


def load_text(docs: List[LangchainDocument]) -> List[LangchainDocument]:
    """Load and process text files.

    Parameters
    ----------
    docs : List[LangchainDocument]
        The list of existing documents to extend.

    Returns
    -------
    List[LangchainDocument]
        The updated list of documents including text files.
    """
    text_loader = TextLoader("../data/01_raw/Apresentação.txt")
    text_documents = text_loader.load()

    for doc in text_documents:
        doc.metadata["source"] = "Apresentação.txt"
    docs.extend(text_documents)
    return docs


def load_json(docs: List[LangchainDocument]) -> List[LangchainDocument]:
    """Load and process JSON files.

    Parameters
    ----------
    docs : List[LangchainDocument]
        The list of existing documents to extend.

    Returns
    -------
    List[LangchainDocument]
        The updated list of documents including JSON files.
    """
    json_path = "../data/01_raw/Exercícios.json"
    data = json.loads(Path(json_path).read_text())

    processed_documents = process_json_data(data)

    with open("../data/02_intermediate/processed_questions.json", "w") as f:
        json.dump([doc.dict() for doc in processed_documents], f)

    json_loader = JSONLoader(
        file_path="../data/02_intermediate/processed_questions.json",
        jq_schema=".",
        text_content=False,
    )
    json_documents = json_loader.load()

    for doc in json_documents:
        doc.metadata["source"] = "Exercícios.json"
    docs.extend(json_documents)
    return docs


def load_image(docs: List[LangchainDocument]) -> List[LangchainDocument]:
    """Load and process image files.

    Parameters
    ----------
    docs : List[LangchainDocument]
        The list of existing documents to extend.

    Returns
    -------
    List[LangchainDocument]
        The updated list of documents including image files.
    """
    image_path = "../data/01_raw/Infografico-1.jpg"
    extracted_text = extract_text_from_image(image_path)

    image_document = LangchainDocument(
        page_content=extracted_text, metadata={"source": "Infografico-1.jpg"}
    )

    docs.append(image_document)
    return docs


def load_video(docs: List[LangchainDocument]) -> List[LangchainDocument]:
    """Load and process video files.

    Parameters
    ----------
    docs : List[LangchainDocument]
        The list of existing documents to extend.

    Returns
    -------
    List[LangchainDocument]
        The updated list of documents including video files.
    """
    video_file = "../data/01_raw/Dica_do_professor.mp4"
    audio_file = "../data/02_intermediate/Dica_do_professor.mp3"

    video = VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)

    config = aai.TranscriptionConfig(language_code="pt")
    audio_loader = AssemblyAIAudioTranscriptLoader(file_path=audio_file, config=config)
    audio_documents = audio_loader.load()

    for doc in audio_documents:
        doc.metadata["source"] = "Dica do professor.mp4"
    docs.extend(audio_documents)
    return docs


def load_data():
    docs = load_pdf()

    docs = load_text(docs)

    docs = load_json(docs)

    docs = load_image(docs)

    docs = load_video(docs)

    # Cleaning metadata
    problematic_docs = find_and_correct_invalid_metadata(docs)

    if problematic_docs:
        logs.info("Documentos com metadados inválidos encontrados e corrigidos:")
        for key, value, doc in problematic_docs:
            logs.info(f"Documento com {key} = {value} foi corrigido")
    else:
        logs.info("Nenhum documento com metadados inválidos encontrado.")

    compress_documents_files = "../data/03_primary/compress_documents.pkl"

    save_to_file(docs, compress_documents_files)

    docs = load_from_file(compress_documents_files)

    chroma_db_dir = "../data/03_primary/chroma_db"
    docsearch = Chroma.from_documents(
        docs,
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory=chroma_db_dir,
    )
    logs.info("Chroma DB salvo com sucesso.")
    return docs, docsearch
