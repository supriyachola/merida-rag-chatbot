
from collections import deque
from urllib.parse import urlparse
import time
import hashlib
import xml.etree.ElementTree as ET

import requests
import chromadb
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_URL = "https://meridatechminds.com"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
MAX_PAGES = 100
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "merida_website"

session = requests.Session()
session.headers.update({
    "User-Agent": "MeridaRAGPrototype/1.0 (educational project)"
})


def get_sitemap_urls():
    response = session.get(SITEMAP_URL, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    urls = []

    for element in root.iter():
        if element.tag.endswith("loc") and element.text:
            url = element.text.strip()
            parsed = urlparse(url)

            if parsed.hostname not in {
                "meridatechminds.com",
                "www.meridatechminds.com",
            }:
                continue

            path = parsed.path.lower()

            # Respect the site's robots.txt restrictions.
            if path.startswith("/admin/") or path.startswith("/api/"):
                continue

            urls.append(url.split("#")[0])

    return list(dict.fromkeys(urls))


def scrape_website():
    urls = get_sitemap_urls()[:MAX_PAGES]
    documents = []

    print(f"Found {len(urls)} public sitemap URLs.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
        )

        page = browser.new_page()
        page.set_default_timeout(15000)

        for index, url in enumerate(urls, start=1):
            try:
                try:
                    page.goto(
                        url,
                        wait_until="networkidle",
                        timeout=30000,
                    )
                except PlaywrightTimeoutError:
                    # Some sites keep network connections open.
                    # Try extracting the page even after a navigation timeout.
                    pass

                page.wait_for_timeout(1000)

                title = page.title() or url

                # Remove repeated navigation and non-content elements.
                page.locator(
                    "nav, footer, script, style, noscript"
                ).evaluate_all(
                    "(elements) => elements.forEach(element => element.remove())"
                )

                main = page.locator("main")
                article = page.locator("article")

                if main.count() > 0:
                    content = main.first.inner_text()
                elif article.count() > 0:
                    content = article.first.inner_text()
                else:
                    content = page.locator("body").inner_text()

                content = " ".join(content.split())

                if len(content) >= 120:
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": url,
                                "title": title,
                            },
                        )
                    )
                    print(
                        f"[{index}/{len(urls)}] "
                        f"Collected {len(content)} characters: {title}"
                    )
                else:
                    print(f"[{index}/{len(urls)}] Little text found: {url}")

            except Exception as error:
                print(f"[{index}/{len(urls)}] Failed: {url} — {error}")

            time.sleep(0.5)

        browser.close()

    return documents


def main():
    documents = scrape_website()

    
    if not documents:
        raise RuntimeError(
            "No page content was extracted. Check the browser and website."
        )

    print(f"\nPages collected before deduplication: {len(documents)}")

    # Remove pages that contain exactly the same text.
    unique_documents = []
    seen_hashes = set()

    for doc in documents:
        normalized_text = " ".join(doc.page_content.split())
        content_hash = hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_documents.append(doc)

    documents = unique_documents

    print(f"Unique pages after deduplication: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    print(f"Text chunks created: {len(chunks)}")
    print("Loading local embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Remove the previous collection on reruns to prevent stale or duplicate data.
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = [
        collection.name for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )

    print("\nIngestion completed successfully!")
    print(f"Pages indexed: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Vector database: {CHROMA_PATH}")


if __name__ == "__main__":
    main()
