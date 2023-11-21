from openai_utils import llm_call
from response_synthesizer.base import BaseResponseSynthesizer

DEFAULT_QA_PROMPT_TEMPLATE = """
We have provided you with some context information and a question below.
Assuming no prior knowledge, answer the question using ONLY the context information provided.
If the answer cannot be obtained from the context, simply reply "I don't know".
Also, briefly justify your answer in one or two sentences.

Here is the context information:
-----------------------------------------------------------
{context}
-----------------------------------------------------------

Please answer the following question:
-----------------------------------------------------------
{question}
-----------------------------------------------------------
"""

class CompactResponseSynthesizer(BaseResponseSynthesizer):
  """Give ALL context information to LLM in one bulk"""
  
  def __init__(self, model = "gpt-3.5-turbo-1106", prompt = DEFAULT_QA_PROMPT_TEMPLATE) -> None:
    super().__init__(model)
    self.prompt = prompt
  
  def synthesize(self, question: str, context: [str]) -> (str, int):
    # we assume that all context chunks can fit in one LLM call
    prompt = self.prompt.format(question = question, context = "\n".join(context))
    response, cost = llm_call(model = self.model, user_prompt = prompt)
    return response["choices"][0]["message"]["content"], cost