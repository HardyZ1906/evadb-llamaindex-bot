from abc import ABC, abstractmethod
from retriever.base import BaseRetriever
from response_synthesizer.base import BaseResponseSynthesizer

class BaseQueryEngine(ABC):
  @abstractmethod
  def __init__(self, retriever: BaseRetriever,
               response_synthesizer: BaseResponseSynthesizer,
               **kwargs) -> None:
    """query engine initializer"""
    self.retriever = retriever
    self.response_synthesizer = response_synthesizer
  
  @abstractmethod
  def query(self, question: str, **kwargs) -> (str, int):
    """answer a question using the given context information"""