from query_engine.base import BaseQueryEngine, BaseRetriever, BaseResponseSynthesizer

class SimpleQueryEngine(BaseQueryEngine):
  def __init__(self, retriever: BaseRetriever,
               response_synthesizer: BaseResponseSynthesizer) -> None:
    super().__init__(retriever, response_synthesizer)
  
  def query(self, question: str) -> (str, int):
    total_cost = 0
    context, cost = self.retriever.retrieve(question)
    total_cost += cost
    
    answer, cost = self.response_synthesizer.synthesize(question, context)
    total_cost += cost
    return answer, cost