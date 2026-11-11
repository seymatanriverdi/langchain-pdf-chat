from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "data" / "denizcilikk.pdf"


loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)
#llm = ChatOllama(
   # model="llama3.2:1b",
    #temperature=0
#)

prompt = ChatPromptTemplate.from_template("""
Sen bir PDF soru-cevap asistanısın.

Görev:
Soruda geçen terimin tanımını bağlamdan bul.

Kurallar:
- Bağlamda "Terim: Tanım" formatı varsa, sadece sorudaki terimin satırını kullan.
- Başka terimlerin tanımlarını kullanma.
- Cevabı kendi bilginle tamamlama.
- Cevap bağlamda yoksa sadece "Bu bilgi dokümanda bulunamadı." de.
- Cevabı Türkçe, kısa ve net ver.

Bağlam:
{context}

Soru:
{question}

Cevap:
""")


while True:
    question = input("\nAsk a question or type 'exit': ")

    if question.lower() == "exit":
        print("Program finished.")
        break

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(final_prompt)

    print("\nRetrieved Context:")
    print(context[:1500])

    print("\nAnswer:")
    print(response.content)