from __future__ import annotations

# 下面俩已经弃用
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 现在使用下面俩
from langchain_pymupdf4llm import PyMuPDF4LLMLoader   # 返回的也是List[Document]
from unstructured.partition.md import partition_md   # 返回的不是List[Document]，需要手动处理

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from typing import List


import re
import os

from app.config import FOLDER_PATH, CHUNK_SIZE, OVERLAP_SIZE


def get_file_paths(folder_path: str = FOLDER_PATH) -> list:
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def process_document(page: Document, file_type: str) -> Document:
    if file_type == 'pdf':
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), page.page_content)
        page.page_content = page.page_content.replace('•', '')
        page.page_content = page.page_content.replace(' ', '')

    elif file_type == 'md':
        page.page_content = page.page_content.replace('\n\n', '\n')

    return page


def load_documents(file_paths: list) -> list:
    documents = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            pdf_document = PyMuPDFLoader(file_path).load()
            documents.extend([process_document(page, 'pdf') for page in pdf_document])

        elif file_path.endswith('.md'):
            md_document = UnstructuredMarkdownLoader(file_path).load()
            documents.extend([process_document(page, 'md') for page in md_document])

    return documents
"""
load()载入后的数据类型变成langchain_core.documents.base.Document，该类型包含两个属性：
    - page_content: 文档的内容
    - metadata: 文档的描述性数据，包含文件名、文件路径、页
"""

def split_documents(pages, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    )
    chunks = text_splitter.split_documents(pages)
    return chunks
"""
切分后还是langchain_core.documents.base.Document类型，只是把每个块变成一个Document了
"""



if __name__ == "__main__":
    file_paths = get_file_paths()
    documents = load_documents(file_paths)

    print(documents[0].page_content[0])
    

    # # 文本分块
    # CHUNK_SIZE = 500
    # OVERLAP_SIZE = 50

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=OVERLAP_SIZE,
    # )

    # chunks = text_splitter.split_documents(documents)







