import os
from datetime import date
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


PDF_CONFIG = [
    {
        "path": "Python.pdf",            
        "source_type": "research_paper",
    },
    {
        "path": "java.pdf",            
        "source_type": "textbook",
    },
]

TEST_QUESTIONS = [
    "What is the main topic discussed in the document?",
    "What are the key findings or conclusions?",
    "Can you summarize the important concepts mentioned?",
]

EMBED_MODEL   = "nomic-embed-text"   
CHAT_MODEL    = "qwen2.5:1.5b"       
CHROMA_DIR    = "./chroma_db"      
TOP_K         = 3                  



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
            print(f"   File not found: {path} — skipping.")
            continue

        print(f"   Loading: {path}")
        loader    = PyPDFLoader(path)
        documents = loader.load()
        chunks    = splitter.split_documents(documents)

        filename  = os.path.basename(path)
        for chunk in chunks:
            chunk.metadata["filename"]    = filename
            chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1
            chunk.metadata["upload_date"] = str(date.today())
            chunk.metadata["source_type"] = config["source_type"]

        all_chunks.extend(chunks)
        print(f"    {len(chunks)} chunks created")

    return all_chunks



def build_vector_store(chunks: list) -> Chroma:
    """
    Generate embeddings using OllamaEmbeddings (nomic-embed-text)
    and store all chunks in a ChromaDB vector store.
    """
    print(f"\n   Generating embeddings using '{EMBED_MODEL}'...")
    print(f"     This may take a minute for large PDFs — please wait...")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vector_store.persist()
    print(f"  {len(chunks)} chunks stored in ChromaDB at '{CHROMA_DIR}'")

    return vector_store


def build_retriever(vector_store: Chroma, top_k: int = TOP_K):
    """
    Create a retriever from ChromaDB that returns the
    top-k most semantically similar chunks for any query.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    print(f"  Semantic retriever ready (top_k={top_k})")
    return retriever


def build_rag_chain(retriever):
    """
    Build a RAG chain:
      question → retriever → prompt (context + question) → LLM → answer
    """

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

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    def format_context(docs) -> str:
        return "\n\n".join(
            f"[Source: {d.metadata.get('filename','?')} | "
            f"Page: {d.metadata.get('page_number','?')}]\n{d.page_content}"
            for d in docs
        )

    rag_chain = (
        {
            "context":  retriever | format_context,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print(f"  RAG chain built with model '{CHAT_MODEL}'")
    return rag_chain, retriever


def test_rag(rag_chain, retriever, questions: list[str]) -> None:
    """
    For each question:
      - Retrieve the top chunks
      - Generate an answer via the RAG chain
      - Print everything clearly
    """
    print("\n" + "═"*62)
    print("  Testing RAG Chain with Questions")
    print("═"*62)

    for i, question in enumerate(questions, start=1):
        print(f"\n{'─'*62}")
        print(f"  Q{i}: {question}")
        print(f"{'─'*62}")

        retrieved_docs = retriever.invoke(question)
        print(f"\n   Retrieved {len(retrieved_docs)} chunk(s):")
        for j, doc in enumerate(retrieved_docs, start=1):
            m = doc.metadata
            print(f"\n    Chunk {j}:")
            print(f"    ├─ File      : {m.get('filename', '?')}")
            print(f"    ├─ Page      : {m.get('page_number', '?')}")
            print(f"    ├─ Type      : {m.get('source_type', '?')}")
            print(f"    └─ Preview   : {doc.page_content[:150].strip()}...")

        print(f"\n  Answer:")
        print(f"  {'─'*56}")
        answer = rag_chain.invoke(question)
        print(f"  {answer.strip()}")
        print(f"  {'─'*56}")

    print(f"\n{'═'*62}")
    print("    All questions tested successfully!")
    print(f"{'═'*62}\n")



def main():
    print("\n" + "═"*62)
    print("   RAG Bootcamp Day 3 — Vector Store + RAG Chain")
    print("   Nunnari Academy")
    print("═"*62)

    print("\n STEP 1: Loading and chunking PDFs...")
    chunks = load_and_chunk_pdfs(PDF_CONFIG)

    if not chunks:
        print("\n No chunks available.")
        print("   Please update PDF_CONFIG with real PDF file paths.\n")
        return

    print(f"\n   Total chunks ready: {len(chunks)}")

    print("\n STEP 2 (Exercise 1): Building ChromaDB vector store...")
    vector_store = build_vector_store(chunks)

    print("\n STEP 3 (Exercise 2): Building semantic retriever...")
    retriever = build_retriever(vector_store)

    print("\n STEP 4 (Exercise 3): Building RAG chain...")
    rag_chain, retriever = build_rag_chain(retriever)

    print("\n STEP 5 (Exercise 4): Testing with questions...")
    test_rag(rag_chain, retriever, TEST_QUESTIONS)


if __name__ == "__main__":
    main()
