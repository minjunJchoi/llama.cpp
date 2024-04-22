from typing import List
from langchain_core.documents import Document

# class Document:
#     def __init__(self, metadata: Dict[str, str], page_content: str):
#         self.metadata = metadata
#         self.page_content = page_content

# 예시 문서들
docs = [
    Document(metadata={"source": "Document 1"}, page_content="Content of Document 1"),
    Document(metadata={"source": "Document 2"}, page_content="Content of Document 2"),
    Document(metadata={"source": "Document 3"}, page_content="Content of Document 3")
]
# citation format
def format_docs(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        doc_str = f"""\
    Content of source ID="{i}": {doc.page_content}
        Origin={doc.metadata['source']}
    """
        formatted.append(doc_str)
    return "\n".join(formatted)
# print(format_docs(retriever.invoke(question)))

# format_docs 함수 실행
formatted_docs = format_docs(docs)
print(formatted_docs)