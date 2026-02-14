import requests
from bs4 import BeautifulSoup
import os
import re
import time
from urllib.parse import urlparse, unquote

# ================= НАСТРОЙКИ =================
URLS_FILE = "links.txt"
OUTPUT_DIR = "raw_text"
MIN_PARAGRAPH_LENGTH = 80  # минимальная длина абзаца
DELAY = 1.5  # задержка между запросами
MAX_RETRIES = 3  # попытки при ошибке
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",

}
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_clean_text(html: str) -> str:
    """Извлекает чистый текст из HTML статьи Wikipedia"""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", id="mw-content-text")
    if not content:
        return ""
    paragraphs = content.find_all("p")
    texts = []
    for p in paragraphs:
        text = p.get_text(" ", strip=True)
        if len(text) >= MIN_PARAGRAPH_LENGTH:
            texts.append(text)
    text = "\n\n".join(texts)
    # удалить ссылочные номера с возможными пробелами, например [ 1 ], [12]
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    # удалить предупреждения о конфликтах источников и другие wiki-метаданные
    text = re.sub(r"(There are two conflicting sources.*See this article's talk page for more information\.)", "", text, flags=re.DOTALL)
    # удалить метки о датах вроде 34 ABY, если они отдельно
    text = re.sub(r"^\d+\s(ABY|BBY)\s*$", "", text, flags=re.MULTILINE)
    # удалить усечения вроде ...(truncated 287366 characters)...
    text = re.sub(r"\.\.\.\(truncated \d+ characters\)\.\.\.", "", text)
    # нормализовать пробелы и переносы
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_text(text: str) -> str:
    """Приводит текст к стандартной форме"""
    return (
        text.replace("–", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace(" , ", ", ")  # нормализовать пробелы вокруг запятых, как в (5 ft , 8 in)
        .strip()
    )

# ---------- ЧТЕНИЕ ССЫЛОК ----------
with open(URLS_FILE, "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
print(f"Найдено ссылок: {len(urls)}")

# ---------- СКАЧИВАНИЕ ЧЕРЕЗ СЕССИЮ ----------
session = requests.Session()
session.headers.update(HEADERS)

for idx, url in enumerate(urls, start=1):
    print(f"[{idx}/{len(urls)}] {url}")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            r.encoding = r.apparent_encoding

            text = extract_clean_text(r.text)
            if not text:
                print("  ⚠ Контент не найден")
                break

            text = normalize_text(text)

            page = unquote(urlparse(url).path.split("/")[-1])
            filename = f"{page}.txt"
            with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
                f.write(text)

            print(f"  ✔ Сохранено ({len(text)} символов)")
            time.sleep(DELAY)
            break  # успех — выходим из retry

        except requests.HTTPError as e:
            if r.status_code == 403:
                print(f"  ✖ Ошибка 403: доступ запрещён. Попытка {attempt}/{MAX_RETRIES}")
                time.sleep(DELAY * 2)
            else:
                print(f"  ✖ HTTP ошибка: {e}")
                break
        except Exception as e:
            print(f"  ✖ Ошибка: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(DELAY)
            else:
                break

print("\nГотово.")