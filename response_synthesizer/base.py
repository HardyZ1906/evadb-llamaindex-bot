from abc import ABC, abstractmethod

class BaseResponseSynthesizer(ABC):
  @abstractmethod
  def __init__(self, model: str, **kwargs) -> None:
    self.model = model
  
  @abstractmethod
  def synthesize(self, question: str, context: [str]) -> (str, int):
    """generate response given the context and the question
    return the answer and the cost"""