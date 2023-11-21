import evadb
from retriever.base import BaseRetriever

class VectorStoreIndexRetriever(BaseRetriever):
  """Text chunks are stored sequentially but have semantics extracted in advance;
  On retrieval, use the top K most relevant chunks as context"""
  
  def __init__(self, cursor: evadb.EvaDBCursor, doc: str, top_k: int = 4) -> None:
    super().__init__(cursor, doc)
    self.top_k = top_k
    self.cursor.query("""
      CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
      IMPL './sentence_feature_extractor.py';
    """).df()
    self.cursor.query(f"""
      CREATE INDEX IF NOT EXISTS {self.doc}_index
      ON {self.doc}(SentenceFeatureExtractor(data))
      USING FAISS;
    """).df()

  def retrieve(self, question: str) -> ([str], int):
    return self.cursor.query(f"""
      SELECT data FROM {self.doc}
      ORDER BY
        Similarity(
          SentenceFeatureExtractor({question}),
          SentenceFeatureExtractor(data)
        )
      LIMIT {self.top_k};
    """).df()[f"{self.doc}.data"].tolist(), 0