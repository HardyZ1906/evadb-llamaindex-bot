from query_engine.base import BaseQueryEngine, BaseRetriever, BaseResponseSynthesizer
import time

class SimpleQueryEngine(BaseQueryEngine):
  def __init__(self, retriever: BaseRetriever,
               response_synthesizer: BaseResponseSynthesizer) -> None:
    super().__init__(retriever, response_synthesizer)
  
  def query(self, question: str) -> (str, int):
    total_cost = 0
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
    return answer, cost