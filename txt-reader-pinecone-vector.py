import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain import VectorDBQA, OpenAI

pinecone.init(api_key="5cfa2038-a280-4426-9b11-1457b14420a4", environment="asia-southeast1-gcp-free")

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader("/home/martins/Documents/AWS BITLOGIC/LangChain/intro-to-vectordb/mediumblogs/mediumblog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    docsearch = Pinecone.from_documents(texts, embeddings, index_name="medium-blogs-embedding-index")

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        vectorstore=docsearch,
        return_source_documents=True
    )

    query = "What is a vector DB? Give me a 15 word answer for a begginner"

    result = qa({"query": query})
    print(result)
