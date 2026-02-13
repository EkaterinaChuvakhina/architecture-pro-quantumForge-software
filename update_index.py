import os
import re
import json
import time
import hashlib
import logging
from datetime import datetime

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import warnings
warnings.filterwarnings("ignore")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from filelock import FileLock, Timeout

EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DATA_DIR = "knowledge_base"
INDEX_DIR = "faiss_index"
STATE_FILE = "faiss_state.json"
LOCK_FILE = "update_faiss.lock"
LOG_FILE = "update_index_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

for name in ["huggingface_hub", "sentence_transformers", "transformers", "torch", "safetensors", "filelock"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def filter_chunk(chunk: str) -> bool:
    dangerous_patterns = [
        r"ignore.*previous.*instructions",
        r"игнорируй.*предыдущие.*инструкции",
        r"system\s*\(\s*'rm\s*-rf",
        r"<python>.*os\.system",
        r"jailbreak",
        r"обход.*защиты"
    ]
    return not any(re.search(pattern, chunk.lower()) for pattern in dangerous_patterns)


def get_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def process_file(filename: str):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )

    file_chunks = splitter.split_text(text)
    docs = []
    doc_ids = []
    for i, chunk in enumerate(file_chunks):
        if filter_chunk(chunk):
            doc_id = f"{filename}__chunk_{i:04d}"
            docs.append(Document(
                page_content=chunk,
                metadata={"source": filename, "chunk_id": i + 1}
            ))
            doc_ids.append(doc_id)
    logger.info(f"Создано {len(docs)} чанков из {filename}")
    return docs, doc_ids


def main():
    start_time = time.time()
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logger.info("=== ЗАПУСК ОБНОВЛЕНИЯ FAISS ИНДЕКСА ===")
    logger.info(f"Время запуска: {start_time_str}")

    state = load_state()
    new_state = {}
    changed_files = []
    removed_files = []
    total_new_chunks = 0
    total_deleted_chunks = 0
    errors_occurred = False

    current_files = set()
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.isfile(filepath) and filename.endswith(".txt"):
            current_files.add(filename)
            file_hash = get_file_hash(filepath)

            if filename in state and state[filename]["hash"] == file_hash:
                new_state[filename] = state[filename].copy()
            else:
                changed_files.append(filename)
                new_state[filename] = {"hash": file_hash, "doc_ids": []}

    removed_files = [f for f in state.keys() if f not in current_files]
    for f in removed_files:
        logger.info(f"→ Удалён файл: {f}")

    if not changed_files and not removed_files:
        logger.info("Изменений не обнаружено. Индекс уже актуален.")
    else:
        logger.info(f"Изменений: {len(changed_files)} (новые/изменённые), {len(removed_files)} (удалённые)")

    try:
        if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
            vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            logger.info("Загружен существующий FAISS индекс")
        else:
            vectorstore = None
            logger.info("Индекс не найден → будет создан новый")
    except Exception as e:
        logger.error("Ошибка при загрузке индекса", exc_info=True)
        errors_occurred = True
        vectorstore = None

    ids_to_delete = []
    for filename in removed_files + changed_files:
        if filename in state and "doc_ids" in state[filename]:
            old_ids = state[filename]["doc_ids"]
            ids_to_delete.extend(old_ids)

    if ids_to_delete and vectorstore:
        try:
            vectorstore.delete(ids_to_delete)
            total_deleted_chunks = len(ids_to_delete)
            logger.info(f"Удалено {total_deleted_chunks} старых чанков (из удалённых/изменённых файлов)")
        except Exception as e:
            logger.error("Ошибка при удалении чанков", exc_info=True)
            errors_occurred = True

    for filename in changed_files:
        try:
            docs, doc_ids = process_file(filename)
            if docs:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(docs, embeddings, ids=doc_ids)
                else:
                    vectorstore.add_documents(docs, ids=doc_ids)
                new_state[filename]["doc_ids"] = doc_ids
                total_new_chunks += len(docs)
                logger.info(f"✓ Добавлено {len(docs)} чанков из {filename}")
        except Exception as e:
            logger.error(f"Ошибка обработки файла {filename}", exc_info=True)
            errors_occurred = True

    try:
        if vectorstore:
            vectorstore.save_local(INDEX_DIR)
            save_state(new_state)
            logger.info("Индекс успешно сохранён")
    except Exception as e:
        logger.error("Ошибка сохранения индекса", exc_info=True)
        errors_occurred = True

    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    duration = time.time() - start_time
    index_size = vectorstore.index.ntotal if vectorstore else 0

    logger.info("=" * 70)
    logger.info("                     ИТОГОВАЯ СВОДКА")
    logger.info("=" * 70)
    logger.info(f"Время запуска          : {start_time_str}")
    logger.info(f"Время завершения       : {end_time_str}")
    logger.info(f"Время работы           : {duration:.2f} секунд")
    logger.info(f"Добавлено новых чанков : {total_new_chunks}")
    logger.info(f"Удалено чанков         : {total_deleted_chunks}")
    logger.info(f"Размер итогового индекса: {index_size} векторов")
    logger.info("СТАТУС: " + ("Успешно завершено" if not errors_occurred else "Завершено С ОШИБКАМИ"))
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        with FileLock(LOCK_FILE, timeout=15):
            main()
    except Timeout:
        logger.warning("Скрипт уже запущен другим процессом. Пропускаем.")
    except Exception as e:
        logger.critical("Критическая ошибка в скрипте", exc_info=True)