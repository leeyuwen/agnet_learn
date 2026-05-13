import os
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import Config


class VectorStore:
    def __init__(self):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        if Config.HF_TOKEN:
            os.environ["HF_TOKEN"] = Config.HF_TOKEN
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"token": Config.HF_TOKEN} if Config.HF_TOKEN else {}
        )
        self.persist_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "chroma_db"
        )
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def create_vector_store(self, documents):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(texts))]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        return self.collection

    def load_vector_store(self):
        return Chroma(
            client=self.client,
            collection_name="documents",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def similarity_search(self, query: str, k: int = 4):
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        from langchain_core.documents import Document
        docs = []
        if results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                docs.append(Document(page_content=doc_text, metadata=metadata))
        return docs

    def clear_index(self):
        try:
            self.client.delete_collection("documents")
        except Exception as e:
            print(f"清空索引失败: {e}")
            return
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents):
        self.create_vector_store(documents)
        return self.collection
