import os
import datetime
import json
from dotenv import load_dotenv
from loguru import logger
from datasets import Dataset

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_gigachat import GigaChat

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
)

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-m3"
INDEX_DIR = "faiss_index"
TOP_K = 10
GIGACHAT_MODEL = "GigaChat"
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")

if not GIGACHAT_CREDENTIALS:
    raise ValueError("GIGACHAT_CREDENTIALS не найден в .env!")

USE_RERANKER = True
RERANKER_MODEL = "BAAI/bge-reranker-large"

logger.info("Загружаю эмбеддинги и FAISS индекс...")
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
        logger.success("Reranker активирован")
    except Exception as e:
        logger.warning(f"Reranker не загрузился: {e}")
        retriever = base_retriever
else:
    retriever = base_retriever

logger.success("Retriever готов")

def get_llm():
    return GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        model=GIGACHAT_MODEL,
        temperature=0.0,
        verify_ssl_certs=False,
        timeout=120
    )

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

def is_dangerous_chunk(text: str) -> bool:
    dangerous_words = {
        "пароль", "password", "root", "secret", "admin", "суперпароль",
        "swordfish", "pwd", "pass", "ключ", "секрет", "superpassword"
    }
    return any(word in text.lower() for word in dangerous_words)

def load_golden_dataset(file_path: str = "golden_questions.txt"):
    questions = []
    ground_truths = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " - " not in line:
                continue
            parts = line.split(" - ", 1)
            question = parts[0].strip()
            if not question.endswith("?"):
                question += "?"
            answer = parts[1].strip()
            questions.append(question)
            ground_truths.append(answer)

    return questions, ground_truths

if __name__ == "__main__":
    logger.info("Загрузка golden questions...")
    questions, ground_truths = load_golden_dataset()

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": ground_truths
    }

    llm = get_llm()

    for idx, question in enumerate(questions):
        logger.info(f"Обрабатываю вопрос {idx+1}/{len(questions)}: {question}")

        docs = retriever.invoke(question)
        original_count = len(docs)
        docs = [doc for doc in docs if not is_dangerous_chunk(doc.page_content)]
        logger.info(f"Найдено чанков: {len(docs)} (отфильтровано {original_count - len(docs)})")

        context_text = "\n\n".join([
            f"Чанк: {doc.page_content}\n(Источник: {doc.metadata.get('source', 'unknown')}, Чанк ID: {doc.metadata.get('chunk_id', 'unknown')})"
            for doc in docs
        ])

        if not docs:
            answer = "В предоставленных материалах нет информации об этом."
        else:
            prompt = SYSTEM_PROMPT.format(docs=context_text, question=question)
            try:
                response = llm.invoke(prompt)
                answer = response.content.strip()
            except Exception as e:
                logger.error(f"Ошибка GigaChat: {e}")
                answer = "В предоставленных материалах нет информации об этом."

        contexts = [doc.page_content for doc in docs]
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)

        import time
        time.sleep(1)

    dataset = Dataset.from_dict(data)

    ragas_llm = get_llm()
    ragas_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logger.info("Запуск оценки Ragas...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ RAGAS")
    print("="*50)
    print(result)
    print("="*50)

    result.to_pandas().to_csv("ragas_evaluation_results.csv", index=False, encoding="utf-8")
    logger.success("Оценка завершена. Результаты сохранены в ragas_evaluation_results.csv")