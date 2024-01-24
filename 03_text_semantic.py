# pip install langchain pypdf redis_server bs4 tiktoken faiss-cpu - https://pypi.org/project/langchain/

from langchain.llms import OpenAI 
from langchain.document_loaders import PyPDFLoader 
from langchain.vectorstores import FAISS, redis
from langchain.embeddings import OpenAIEmbeddings 
from langchain.chains import ConversationalRetrievalChain 
from langchain.memory import ConversationBufferMemory 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os
import getpass
import PyPDF2
# import subprocess
# import redis_server

# create loader pdf with pyPDF2
def load_pdf_2(path):
    '''
    Cargamos el PDF y lo separamos en páginas
    '''
    pdfFileObj = open(path, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    pages = []
    for i in range(len(pdfReader.pages)):
        page = pdfReader.pages[i]
        # pages.append(page.extract_text())
        pages.append(page)
    return pages

# create loader pdf with pypdfloader
def load_pdf(path):
    '''
    Cargamos el PDF y lo separamos en páginas
    '''
    # print(PyPDFLoader(path).load())
    return PyPDFLoader(path).load_and_split()


# text splitter
def split_text(pages):
    '''
    Fragmentamos las paginas en secciones de 1000 caracteres con un overlap de 100 caracteres entre secciones
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_documents(pages)


# create faiss index from text sections and embeddings from openai api
def create_index(sections):
    '''
    Creamos el index faiss con las secciones del documento y los embeddings de OpenAI
    '''
    # create index with embeddings and sections of text 
    index = FAISS.from_documents(sections, OpenAIEmbeddings(client=OpenAI(), model='text-embedding-ada-002'))
    # FAISS.load_local('index.faiss')
    # index = FAISS.from_texts(sections, OpenAIEmbeddings(client=OpenAI(), model='text-embedding-ada-002'))
    

    # create index with embeddings loaded from redis
    # index = FAISS.from_documents(sections, redis.RedisVectorStore(redis_client, 'embeddings'))
    
    # index = FAISS(sections, embeddings, 'Flat', 512)
    return index


# define conversational chain with memory and index faiss
def create_chain(index):
    '''
    Creamos la cadena conversacional con la memoria y el index
    '''
    # create memory
    retriever = index.as_retriever()
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer'
    )

    # create chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(),
        retriever=retriever,
        memory=memory
    )

    return chain


if __name__ == "__main__":
    # Inicializamos el servidor de redis (Google colab)
    # subprocess.Popen([redis_server.REDIS_SERVER_PATH])

    # Step 2: Set the OpenAI API key
    os.environ["OPENAI_API_KEY"] = "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"

    # load pdf
    pages = load_pdf(os.path.join(os.getcwd(), "files", "acta_test.pdf"))
    #pages = load_pdf_2(os.path.join(os.getcwd(), "files", "16736_ACT_CON.pdf"))
    #print(pages)
    sections = split_text(pages)
    index = create_index(sections)
    chain = create_chain(index)

    print(f'Bienvenido, ¿cual es tu pregunta?')
    while True:
        try:
            user_input = input('Q: ')
            answer = chain({'question': user_input})
            # print(answer)
            print(f"A: {answer['answer'].strip()}")
        except EOFError:
            break
        except KeyboardInterrupt:
            break

    print("Adios!")