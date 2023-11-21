from query_engine.base import BaseQueryEngine, BaseRetriever, BaseResponseSynthesizer
from evaluator import Evaluator

class RetryQueryEngine:
  def __init__(self, retriever: BaseRetriever,
               response_synthesizer: BaseResponseSynthesizer,
               evaluator: Evaluator,
               max_retries: int = 5) -> None:
    super().__init__(retriever, response_synthesizer)
    self.evaluator = evaluator
    self.max_retries = max_retries
  
  def query(self, question: str) -> (str, int):
    total_cost = 0
    for _ in range(self.max_retries):
      context, cost = self.retriever.retrieve(question)
      total_cost += cost
      
      answer, cost = self.response_synthesizer.synthesize(question, context)
      total_cost += cost
      
      result, cost = self.evaluator.evaluate(context, question, answer)
      total_cost += cost
      if result.passing:
        return answer, total_cost