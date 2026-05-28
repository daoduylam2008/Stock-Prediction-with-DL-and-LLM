import warnings
from typing import List

warnings.filterwarnings("ignore")

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from RAG.embeddings import TFIDFEmbeddings


def build_vectorstore(articles, prices, fundamentals, ticker) -> FAISS:
    docs: List[Document] = []

    for art in articles:
        docs.append(Document(
            page_content=(f"HEADLINE: {art['headline']}\n"
                          f"SUMMARY: {art['summary']}\n"
                          f"SOURCE: {art['source']}\nPUBLISHED: {art['published']}"),
            metadata={"type": "news", "ticker": ticker, "headline": art["headline"]},
        ))

    for i in range(0, len(prices), 5):
        chunk = prices[i:i+5]
        lines = [f"{r['date']}: O={r['open']} C={r['close']} H={r['high']} L={r['low']} V={r['volume']}"
                 for r in chunk]
        docs.append(Document(
            page_content="PRICE DATA:\n" + "\n".join(lines),
            metadata={"type": "price", "ticker": ticker},
        ))

    fund_text = "\n".join(f"{k}: {v}" for k, v in fundamentals.items())
    docs.append(Document(
        page_content=f"FUNDAMENTALS FOR {ticker}:\n{fund_text}",
        metadata={"type": "fundamentals", "ticker": ticker},
    ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    return FAISS.from_documents(split_docs, TFIDFEmbeddings())


def query(vs: FAISS, question: str, k: int = 4) -> str:
    docs = vs.similarity_search(question, k=k)
    return "\n---\n".join(d.page_content for d in docs)