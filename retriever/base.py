from abc import ABC, abstractmethod
import evadb

class BaseRetriever(ABC):
  @abstractmethod
  def __init__(self, cursor: evadb.EvaDBCursor, doc: str, **kwargs) -> None:
    """retriever initializer"""
    self.cursor = cursor
    self.doc = doc

  @abstractmethod
  def retrieve(self, question: str) -> ([str], int):
    """retrieve relevant context for a given question"""