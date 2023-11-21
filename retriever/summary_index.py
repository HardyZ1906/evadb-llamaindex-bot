import evadb
from retriever.base import BaseRetriever

class SummaryIndexRetriever(BaseRetriever):
  """Text chunks are stored sequentially;
  On retrieval, simply use ALL text chunks as context information"""
  
  def __init__(self, cursor: evadb.EvaDBCursor, doc: str) -> None:
    super().__init__(cursor, doc)

  def retrieve(self, question: str) -> ([str], int):
    return self.cursor.query(f"""
      SELECT data FROM {self.doc};
    """).df()[f"{self.doc}.data"].tolist(), 0
