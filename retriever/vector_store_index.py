import evadb
from retriever.base import BaseRetriever

class VectorStoreIndexRetriever(BaseRetriever):
  """Text chunks are stored sequentially, but with semantic features extracted in advance;
  On retrieval, use the top K most relevant chunks as context"""
  
  def __init__(self, cursor: evadb.EvaDBCursor, doc: str, top_k: int = 4, new: bool = False) -> None:
    super().__init__(cursor, doc)
    self.top_k = top_k
    if new:
      self.cursor.query("""
        CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
        IMPL './sentence_feature_extractor.py';
      """).df()
      self.cursor.query(f"""
        DROP TABLE IF EXISTS {self.doc}_features;
      """).df()
      self.cursor.query(f"""
        CREATE TABLE {self.doc}_features AS
        SELECT SentenceFeatureExtractor(data), chunk_id
        FROM {self.doc};
      """).df()
      cursor.query(f"""
        DROP INDEX IF EXISTS {self.doc}_features_index;
      """).df()
      cursor.query(f"""
        CREATE INDEX {self.doc}_features_index
        ON {self.doc}_features (features)
        USING FAISS;
      """).df()
    # self.cursor.query(f"""
    #   CREATE INDEX IF NOT EXISTS {self.doc}_index
    #   ON {self.doc}(SentenceFeatureExtractor(data))
    #   USING FAISS;
    # """).df()

  def retrieve(self, question: str) -> ([str], int):
    chunk_ids = self.cursor.query(f"""
      SELECT chunk_id FROM {self.doc}_features
      ORDER BY
        Similarity(
          SentenceFeatureExtractor("{question}"),
          features
        )
      LIMIT {self.top_k};
    """).df()[f"{self.doc}_features.chunk_id"].tolist()
    chunks = []
    for chunk_id in chunk_ids:
      chunks.append(self.cursor.query(f"""
        SELECT data FROM {self.doc} WHERE chunk_id = {chunk_id};
      """).df()[f"{self.doc}.data"][0])
    # print("\n\n\n".join(chunks))
    
    return chunks, 0