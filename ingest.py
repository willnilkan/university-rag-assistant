
from pathlib import Path
import hashlib
import re

# pyrefly: ignore[missing-import]
from langchain_community.document_loaders import PyMuPDFLoader
# pyrefly: ignore[missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore[missing-import]
from langchain_ollama import OllamaEmbeddings
# pyrefly: ignore[missing-import]
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "chroma_db"
COLLECTION = "university_knowledge"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_vectorstore():
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b"
    )

    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(DB_PATH)
    )


def ingest_pdf(uploaded_file):
    """Save and index a PDF uploaded through Streamlit."""

    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")

    # Use a content hash to identify duplicate documents
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Clean the original filename
    filename = Path(uploaded_file.name).name
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

    # Unique filename avoids overwriting other PDFs
    saved_name = f"{file_hash[:12]}_{filename}"
    pdf_path = DATA_DIR / saved_name

    vectorstore = get_vectorstore()

    # Check whether this exact PDF has already been indexed
    existing = vectorstore.get(
        where={"file_hash": file_hash},
        limit=1
    )

    if existing["ids"]:
        return f"Already indexed: {filename}"

    # Save PDF to data folder
    pdf_path.write_bytes(pdf_bytes)

    try:
        # Load PDF
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No extractable text found in PDF.")

        # Attach source information to every chunk
        for chunk in chunks:
            chunk.metadata["file_hash"] = file_hash
            chunk.metadata["filename"] = filename
            chunk.metadata["source"] = saved_name

        # Stable IDs prevent duplicate entries
        chunk_ids = [
            f"{file_hash}_{i}"
            for i in range(len(chunks))
        ]

        vectorstore.add_documents(
            documents=chunks,
            ids=chunk_ids
        )

        return (
            f"Successfully indexed {filename} "
            f"({len(chunks)} chunks)"
        )

    except Exception:
        # Remove any partially indexed chunks
        partial = vectorstore.get(
            where={"file_hash": file_hash}
        )
        if partial["ids"]:
            vectorstore.delete(ids=partial["ids"])

        # Remove PDF if ingestion failed
        pdf_path.unlink(missing_ok=True)
        raise


def list_documents():
    """Return the PDFs stored in the data folder."""
    return sorted(DATA_DIR.glob("*.pdf"))
