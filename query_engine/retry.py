from query_engine.base import BaseQueryEngine, BaseRetriever, BaseResponseSynthesizer
from evaluator import Evaluator
import time

class RetryQueryEngine(BaseQueryEngine):
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
      t1 = time.clock_gettime(time.CLOCK_REALTIME)
      context, cost = self.retriever.retrieve(question)
      t2 = time.clock_gettime(time.CLOCK_REALTIME)
      print(f"context retrieval time: {t2 - t1}")
      total_cost += cost
      
      t1 = time.clock_gettime(time.CLOCK_REALTIME)
      answer, cost = self.response_synthesizer.synthesize(question, context)
      t2 = time.clock_gettime(time.CLOCK_REALTIME)
      print(f"response synthesis time: {t2 - t1}")
      total_cost += cost
      
      t1 = time.clock_gettime(time.CLOCK_REALTIME)
      result, cost = self.evaluator.evaluate(context, question, answer)
      t2 = time.clock_gettime(time.CLOCK_REALTIME)
      print(f"evaluation time: {t2 - t1}")
      total_cost += cost
      if result.passing:
        break
    return answer, total_cost