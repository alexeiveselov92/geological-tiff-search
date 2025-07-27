import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TESSERACT_LANGUAGES = "rus"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

MAX_TOKENS_CONTEXT = 8000

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_MODEL = "ai-forever/sbert_large_nlu_ru"

DATA_PATHS = {
    "tiff_reports": "tiff_reports/",
    "test_files": "test_files/",
    "extracted_text": "data/extracted_text/",
    "processed_chunks": "data/processed_chunks/",
    "embeddings": "data/embeddings/",
}
