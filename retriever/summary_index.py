import evadb

class SummaryIndex:
  """Text chunks are stored sequentially;
  On retrieval, simply use ALL text chunks as context information"""
  
  def __init__(self, cursor: evadb.EvaDBCursor, doc: str) -> None:
    self.cursor = cursor
    self.doc = doc

  def retrieve(self) -> [str]:
    return self.cursor.query(f"""
      SELECT data FROM {self.doc};
    """).df()["data"].tolist()
