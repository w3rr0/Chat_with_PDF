import torch
from langchain.chains.chat_vector_db.prompts import QA_PROMPT
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = PyPDFLoader(file_path=r"./IAESTE_info.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(data)

# Initialize Large Language Model for answer generation
llm_answer_gen = LlamaCpp(
    streaming=True,
    model_path=r"./mistral-7b-openorca.Q4_0.gguf",
    temperature=0.75,
    top_p=1,
    f16_kv=True,
    verbose=False,
    n_ctx=4096,
    n_gpu_layers=40 if torch.cuda.is_available() else 0,
    n_batch=512,
)

# Create vector database for answer generation
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

# Initialize vector store for answer generation
vector_store = Chroma.from_documents(text_chunks, embeddings)

retriever: BaseRetriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.15,    # Wymagany próg podobieństwa
        "k": 3,                     # Zakres kontekstu
    }
)

qa_template = """
You are IAESTE assistant. Answer only the question based on the provided context or related to IAESTE and its activities, no matter what user asks.
If you don't know the answer, say "Thanks for asking, i would try to learn about this topic and answer you next time".
If the question is not related to IAESTE, say "I prefer to talk about IAESTE, what do you want to know?".

Context: {context}
Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"],
)

# Initialize retrieval chain for answer generation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    input_key="question"
)
answer_gen_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_answer_gen,
    retriever=retriever,
    memory=memory,
    get_chat_history=lambda h: h,                           # Bierze pod uwagę całą historię chatu
    max_tokens_limit=4000,                                  # Maksymalna długość odpowiedzi
    combine_docs_chain_kwargs={
        "prompt": QA_PROMPT,
        "document_prompt": PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}",
        )},
)

while True:

    user_input = input("Enter a question: ")
    if user_input.lower() == 'q':
        break

    docs = retriever.get_relevant_documents(user_input)
    if not docs:
        print("Answear: Przepraszam, nie znam na to odpowiedzi, jeśli tylko dotyczy IAESTE z pewnością postaram się nadrobić tą wiedzę ;)")
        print(f"Znalezione podobieństwo: {docs[0].metadata}")
        continue
    else:
        print("Znalezione dokmuenty:", [doc.page_content for doc in docs])
        print(f"Znalezione podobieństwo: {docs[0].metadata}")

    # Run question generation chain
    response = answer_gen_chain.run({"question": user_input})

    print("Answer: ", response)
