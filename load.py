import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def load_pdfs(pdf_folder: str) -> list:
    """Load all PDFs from a folder with per-file error handling."""
    # Fix 4: Guard against missing folder
    if not os.path.isdir(pdf_folder):
        print(f"Error: folder '{pdf_folder}' does not exist.")
        sys.exit(1)

    all_documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            # Fix 5: Per-file error handling so one bad PDF doesn't crash everything
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                all_documents.extend(documents)
                print(f"Loaded: {file}")
            except Exception as e:
                print(f"Warning: skipping '{file}' — {e}")

    return all_documents


def main():
    # =========================
    # LOAD ALL PDFs
    # =========================
    pdf_folder = "data"
    all_documents = load_pdfs(pdf_folder)

    print(f"\nTotal pages loaded: {len(all_documents)}")

    # Fix 6: Guard against empty document list before building vectorstore
    if not all_documents:
        print("No PDF pages loaded. Add PDFs to the 'data/' folder and try again.")
        sys.exit(1)

    # =========================
    # SPLIT TEXT INTO CHUNKS
    # =========================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(all_documents)
    print(f"\nCreated {len(docs)} chunks")

    # =========================
    # CREATE EMBEDDINGS
    # =========================
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # =========================
    # CREATE VECTOR DATABASE
    # =========================
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("\nFAISS vector database created")

    # =========================
    # LOAD LOCAL GEMMA MODEL
    # =========================
    llm = OllamaLLM(model="gemma:2b")

    # =========================
    # CREATE RETRIEVER
    # =========================
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # =========================
    # CREATE RAG CHAIN (modern LCEL style — no RetrievalQA)
    # =========================
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know the answer, say "I don't know."

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # =========================
    # CHAT LOOP
    # =========================
    print("\nRAG Chatbot Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask Question: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        try:
            response = qa_chain.invoke(query)
            print("\nAnswer:\n")
            print(response)
        except Exception as e:
            print(f"Error during query: {e}")

        print("\n" + "=" * 50 + "\n")


# Fix 8: Guard so the script only runs when executed directly
if __name__ == "__main__":
    main()
