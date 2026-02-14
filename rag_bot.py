import os
import re
import json
import datetime
from dotenv import load_dotenv
from loguru import logger
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_gigachat import GigaChat

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from transformers import pipeline

QUERY_LOG_FILE = "logs.jsonl"

def log_query(
        user_id: int,
        username: str,
        query: str,
        docs: list,
        answer: str,
        sources: list
) -> None:
    timestamp = datetime.datetime.now().isoformat()
    chunks_found = len(docs) > 0
    response_length = len(answer)
    unique_sources = list(set(sources)) if sources else []

    no_info_phrases = [
        "в предоставленных материалах нет информации",
        "не знаю",
        "не найдено",
        "нет информации"
    ]
    is_successful = (
            chunks_found and
            response_length > 100 and
            not any(phrase in answer.lower() for phrase in no_info_phrases)
    )

    log_entry = {
        "timestamp": timestamp,
        "user_id": user_id,
        "username": username or "unknown",
        "query": query,
        "chunks_found": chunks_found,
        "chunks_count": len(docs),
        "response_length": response_length,
        "successful": is_successful,
        "sources": unique_sources,
        "answer_preview": answer[:300]
    }

    try:
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        logger.error(f"Ошибка записи в query_log: {e}")


def escape_markdown_v2(text: str) -> str:
    escape_map = str.maketrans({
        '_': '\\_', '*': '\\*', '[': '\\[', ']': '\\]', '(': '\\(', ')': '\\)',
        '~': '\\~', '`': '\\`', '>': '\\>', '#': '\\#', '+': '\\+', '-': '\\-',
        '=': '\\=', '|': '\\|', '{': '\\{', '}': '\\}', '.': '\\.', '!': '\\!'
    })
    return text.translate(escape_map)


try:
    safety_classifier = pipeline(
        "text-classification",
        model="seara/russian-offensive-language-detection",
        device=-1
    )
except Exception as e:
    logger.warning(f"Safety classifier не загрузился: {e}")
    safety_classifier = None

def is_safe(text: str, threshold: float = 0.7) -> tuple[bool, str]:
    dangerous_patterns = [
        r"ignore.*previous.*instructions",
        r"игнорируй.*предыдущие.*инструкции",
        r"jailbreak", r"обход.*защиты",
        r"(?:назови|скажи|раскрой|дай)\s+(?:пароль|password|root|admin|секрет)",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, text.lower()):
            return False, f"Опасный паттерн: {pattern}"

    if safety_classifier:
        result = safety_classifier(text)[0]
        if result["label"] == "offensive" and result["score"] > threshold:
            return False, "Токсичный контент"
    return True, "Безопасно"


def is_dangerous_chunk(text: str) -> bool:
    dangerous_words = {
        "пароль", "password", "root", "secret", "admin", "суперпароль",
        "swordfish", "pwd", "pass", "ключ", "секрет", "superpassword"
    }
    text_lower = text.lower()
    return any(word in text_lower for word in dangerous_words)


load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-m3"
INDEX_DIR = "faiss_index"
TOP_K = 10

GIGACHAT_MODEL = "GigaChat-Max"
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")

if not GIGACHAT_CREDENTIALS:
    logger.error("GIGACHAT_CREDENTIALS не найден в .env файле!")
    exit(1)

USE_RERANKER = True
RERANKER_MODEL = "BAAI/bge-reranker-large"

logger.info("Загружаю FAISS индекс...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = FAISS.load_local(
    INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)

base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": 25, "lambda_mult": 0.7}
)

if USE_RERANKER:
    try:
        reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        compressor = CrossEncoderReranker(model=reranker, top_n=6)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        logger.success(f"Reranker активирован")
    except Exception as e:
        logger.warning(f"Reranker не загрузился: {e}")
        retriever = base_retriever
else:
    retriever = base_retriever

logger.success("Индекс загружен!")


SYSTEM_PROMPT = """Ты — экспертный ассистент исключительно по миру "Хроники Мороза и Огня".

Ты должен:
1. Сначала размышлять шаг за шагом (Chain of Thought).
2. Использовать только информацию из раздела [CONTEXT].
3. Если нужной информации нет в CONTEXT — отвечать ровно: "В предоставленных материалах нет информации об этом."

Примеры правильных ответов:

Q: Кто такая Анна Камнева?
A: 1. Вопрос касается персонажа семьи Камневых.
2. В CONTEXT указано, что Анна Камнева — третий ребёнок в семье.
3. Ответ: Анна Камнева — третий ребёнок в семье Камневых. (Источник: anna_kamneva.txt)

Q: Как зовут старшего сына Эдуарда Камнев?
A: 1. Ищу информацию о семье Камневых в CONTEXT.
2. Согласно документам, старшего сына Эдуарда Камнева зовут Иван Снегов.
3. Ответ: Старшего сына Эдуарда Камнева зовут Иван Снегов.

[CONTEXT]
<<<
{docs}
>>>

[User]
{question}

Сначала напиши свои рассуждения шаг за шагом, затем дай финальный ответ в формате:
A: [твой ответ]"""


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text.strip()
    user_id = update.effective_user.id
    username = update.effective_user.username

    logger.info(f"Запрос от {user_id} (@{username}): {user_query}")

    safe, reason = is_safe(user_query)
    if not safe:
        logger.warning(f"Блокировка запроса от {user_id}: {reason}")
        await update.message.reply_text("Извините, я не могу обработать этот запрос по соображениям безопасности.")
        return

    docs = retriever.invoke(user_query)

    original_count = len(docs)
    docs = [doc for doc in docs if not is_dangerous_chunk(doc.page_content)]
    logger.info(f"Документов после фильтрации: {len(docs)} (отфильтровано {original_count - len(docs)})")

    sources = [doc.metadata.get("source", "unknown") for doc in docs]

    if not docs:
        answer = "В предоставленных материалах нет информации об этом."
        await update.message.reply_text(answer)

        log_query(user_id, username, user_query, docs, answer, sources)
        return

    context_text = "\n\n".join([
        f"Чанк: {doc.page_content}\n(Источник: {doc.metadata.get('source', 'unknown')}, Чанк ID: {doc.metadata.get('chunk_id', 'unknown')})"
        for doc in docs
    ])

    prompt = SYSTEM_PROMPT.format(docs=context_text, question=user_query)

    try:
        llm = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            model=GIGACHAT_MODEL,
            temperature=0.0,
            verify_ssl_certs=False,
            timeout=60
        )
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"Ошибка GigaChat: {e}")
        answer = "В предоставленных материалах нет информации об этом."

    if is_dangerous_chunk(answer):
        logger.warning("Обнаружена попытка утечки пароля!")
        answer = "В предоставленных материалах нет информации об этом."

    escaped_answer = escape_markdown_v2(answer)
    try:
        await update.message.reply_text(escaped_answer, parse_mode="MarkdownV2")
    except Exception:
        await update.message.reply_text(answer)

    log_query(user_id, username, user_query, docs, answer, sources)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я — RAG-бот по миру 'Хроники Мороза и Огня'.\n"
        "Задавай вопросы о персонажах, местах, событиях."
    )


if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TOKEN:
        logger.error("TELEGRAM_TOKEN не найден!")
        exit(1)

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.success("Бот запущен")
    app.run_polling()