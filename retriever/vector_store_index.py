import evadb

class VectorStoreIndex:
  """Text chunks are stored sequentially but have semantics extracted in advance;
  On retrieval, use the top K most relevant chunks as context"""
  
  def __init__(self, cursor: evadb.EvaDBCursor, doc: str) -> None:
    self.doc = doc
    self.cursor = cursor
    self.cursor.query("""
      CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
      IMPL './sentence_feature_extractor.py';
    """).df()
    self.cursor.query(f"""
      CREATE INDEX IF NOT EXISTS {self.doc}_index
      ON {self.doc}(SentenceFeatureExtractor(data))
      USING FAISS;
    """).df()

  def retrieve(self, question: str, top_k: int = 4) -> [str]:
    return self.cursor.query(f"""
      SELECT data FROM {self.doc}
      ORDER BY
        Similarity(
          SentenceFeatureExtractor({question}),
          SentenceFeatureExtractor(data)
        )
      LIMIT {top_k};
    """).df()["data"].tolist()