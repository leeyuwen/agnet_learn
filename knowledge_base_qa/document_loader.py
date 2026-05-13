import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config


class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )

    def load_single_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        return loader.load()

    def load_folder(self, folder_path: str = None):
        if folder_path is None:
            folder_path = Config.DOCS_FOLDER

        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        all_docs = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    docs = self.load_single_file(file_path)
                    for doc in docs:
                        doc.metadata["source"] = file_path
                    all_docs.extend(docs)
                    print(f"加载成功: {file_path}")
                except Exception as e:
                    print(f"加载失败 {file_path}: {e}")

        return all_docs

    def split_documents(self, documents):
        return self.text_splitter.split_documents(documents)

    def load_and_split(self, folder_path: str = None):
        documents = self.load_folder(folder_path)
        return self.split_documents(documents)
