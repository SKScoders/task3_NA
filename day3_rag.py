"""
╔══════════════════════════════════════════════════════════════╗
║         RAG Bootcamp — Day 3 Task                            ║
║         Vector Store, Semantic Retriever & RAG Chain         ║
║         Nunnari Academy                                       ║
╚══════════════════════════════════════════════════════════════╝

Exercises Covered:
  1. Store chunks in ChromaDB using OllamaEmbeddings (nomic-embed-text)
  2. Build a semantic retriever (top-k similarity search)
  3. Build a RAG chain with prompt template + ChatOllama (qwen2.5:1.5b)
  4. Test with 3+ different questions and print results
"""

import os
from datetime import date

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


# ─────────────────────────────────────────────────────────────
# Configuration — update PDF paths to your actual files
# ─────────────────────────────────────────────────────────────

PDF_CONFIG = [
    {
        "path": "paper1.pdf",            # ← your PDF file name
        "source_type": "research_paper",
    },
    {
        "path": "paper2.pdf",            # ← your PDF file name
        "source_type": "textbook",
    },
]

# Test questions — update these to match your PDF content
TEST_QUESTIONS = [
    "What is the main topic discussed in the document?",
    "What are the key findings or conclusions?",
    "Can you summarize the important concepts mentioned?",
]

EMBED_MODEL   = "nomic-embed-text"   # OllamaEmbeddings model
CHAT_MODEL    = "qwen2.5:1.5b"       # ChatOllama model
CHROMA_DIR    = "./chroma_db"        # folder to persist ChromaDB
TOP_K         = 3                    # number of chunks to retrieve


# ─────────────────────────────────────────────────────────────
# Reused from Day 2 — Load, Split, Attach Metadata
# ─────────────────────────────────────────────────────────────

def load_and_chunk_pdfs(pdf_configs: list[dict]) -> list:
    """Load PDFs, split into chunks, and attach metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    all_chunks = []

    for config in pdf_configs:
        path = config["path"]

        if not os.path.exists(path):
            print(f"  ⚠️  File not found: {path} — skipping.")
            continue

        print(f"  📄 Loading: {path}")
        loader    = PyPDFLoader(path)
        documents = loader.load()
        chunks    = splitter.split_documents(documents)

        # Attach metadata (Day 2 exercise)
        filename  = os.path.basename(path)
        for chunk in chunks:
            chunk.metadata["filename"]    = filename
            chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1
            chunk.metadata["upload_date"] = str(date.today())
            chunk.metadata["source_type"] = config["source_type"]

        all_chunks.extend(chunks)
        print(f"     ✅ {len(chunks)} chunks created")

    return all_chunks


# ─────────────────────────────────────────────────────────────
# Exercise 1 — Store Chunks in ChromaDB
# ─────────────────────────────────────────────────────────────

def build_vector_store(chunks: list) -> Chroma:
    """
    Generate embeddings using OllamaEmbeddings (nomic-embed-text)
    and store all chunks in a ChromaDB vector store.
    """
    print(f"\n  🔢 Generating embeddings using '{EMBED_MODEL}'...")
    print(f"     This may take a minute for large PDFs — please wait...")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Store in ChromaDB (persisted to disk at CHROMA_DIR)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vector_store.persist()
    print(f"  ✅ {len(chunks)} chunks stored in ChromaDB at '{CHROMA_DIR}'")

    return vector_store


# ─────────────────────────────────────────────────────────────
# Exercise 2 — Build a Semantic Retriever
# ─────────────────────────────────────────────────────────────

def build_retriever(vector_store: Chroma, top_k: int = TOP_K):
    """
    Create a retriever from ChromaDB that returns the
    top-k most semantically similar chunks for any query.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    print(f"  ✅ Semantic retriever ready (top_k={top_k})")
    return retriever


# ─────────────────────────────────────────────────────────────
# Exercise 3 — Build the RAG Chain
# ─────────────────────────────────────────────────────────────

def build_rag_chain(retriever):
    """
    Build a RAG chain:
      question → retriever → prompt (context + question) → LLM → answer
    """

    # Prompt template with {context} and {question}
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use the following context
extracted from documents to answer the question clearly and accurately.
If the answer is not in the context, say "I don't have enough information
in the provided documents to answer this."

Context:
{context}

Question:
{question}

Answer:"""
    )

    # ChatOllama LLM
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    # Helper to format retrieved chunks into a single string
    def format_context(docs) -> str:
        return "\n\n".join(
            f"[Source: {d.metadata.get('filename','?')} | "
            f"Page: {d.metadata.get('page_number','?')}]\n{d.page_content}"
            for d in docs
        )

    # RAG chain using LangChain's LCEL (pipe syntax)
    rag_chain = (
        {
            "context":  retriever | format_context,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print(f"  ✅ RAG chain built with model '{CHAT_MODEL}'")
    return rag_chain, retriever


# ─────────────────────────────────────────────────────────────
# Exercise 4 — Test with Multiple Questions
# ─────────────────────────────────────────────────────────────

def test_rag(rag_chain, retriever, questions: list[str]) -> None:
    """
    For each question:
      - Retrieve the top chunks
      - Generate an answer via the RAG chain
      - Print everything clearly
    """
    print("\n" + "═"*62)
    print("  🧪  Testing RAG Chain with Questions")
    print("═"*62)

    for i, question in enumerate(questions, start=1):
        print(f"\n{'─'*62}")
        print(f"  Q{i}: {question}")
        print(f"{'─'*62}")

        # Retrieved chunks
        retrieved_docs = retriever.invoke(question)
        print(f"\n  📚 Retrieved {len(retrieved_docs)} chunk(s):")
        for j, doc in enumerate(retrieved_docs, start=1):
            m = doc.metadata
            print(f"\n    Chunk {j}:")
            print(f"    ├─ File      : {m.get('filename', '?')}")
            print(f"    ├─ Page      : {m.get('page_number', '?')}")
            print(f"    ├─ Type      : {m.get('source_type', '?')}")
            print(f"    └─ Preview   : {doc.page_content[:150].strip()}...")

        # Generated answer
        print(f"\n  🤖 Answer:")
        print(f"  {'─'*56}")
        answer = rag_chain.invoke(question)
        print(f"  {answer.strip()}")
        print(f"  {'─'*56}")

    print(f"\n{'═'*62}")
    print("  ✅  All questions tested successfully!")
    print(f"{'═'*62}\n")


# ─────────────────────────────────────────────────────────────
# Main — wire everything together
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*62)
    print("   RAG Bootcamp Day 3 — Vector Store + RAG Chain")
    print("   Nunnari Academy")
    print("═"*62)

    # ── Load & chunk PDFs (reused from Day 2) ─────────────────
    print("\n📂 STEP 1: Loading and chunking PDFs...")
    chunks = load_and_chunk_pdfs(PDF_CONFIG)

    if not chunks:
        print("\n❌ No chunks available.")
        print("   Please update PDF_CONFIG with real PDF file paths.\n")
        return

    print(f"\n  📦 Total chunks ready: {len(chunks)}")

    # ── Exercise 1: Build ChromaDB vector store ───────────────
    print("\n🗄️  STEP 2 (Exercise 1): Building ChromaDB vector store...")
    vector_store = build_vector_store(chunks)

    # ── Exercise 2: Build semantic retriever ──────────────────
    print("\n🔍 STEP 3 (Exercise 2): Building semantic retriever...")
    retriever = build_retriever(vector_store)

    # ── Exercise 3: Build RAG chain ───────────────────────────
    print("\n⛓️  STEP 4 (Exercise 3): Building RAG chain...")
    rag_chain, retriever = build_rag_chain(retriever)

    # ── Exercise 4: Test with questions ───────────────────────
    print("\n🧪 STEP 5 (Exercise 4): Testing with questions...")
    test_rag(rag_chain, retriever, TEST_QUESTIONS)


if __name__ == "__main__":
    main()
