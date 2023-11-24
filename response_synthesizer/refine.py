from openai_utils import llm_call
from response_synthesizer.base import BaseResponseSynthesizer
from response_synthesizer.compact import DEFAULT_QA_PROMPT_TEMPLATE

DEFAULT_REFINE_PROMPT_TEMPLATE = """
Below is a question:
-----------------------------------------------------------
{question}
-----------------------------------------------------------

We already have an existing answer to this question:
-----------------------------------------------------------
{answer}
-----------------------------------------------------------

Besides, we also have some additional context:
-----------------------------------------------------------
{context}
-----------------------------------------------------------

Assuming no prior knowledge and based on the additional context ONLY,
please refine the original answer (only if needed).
Reply the refined answer. If the original answer is already good enough, or
if the additional context is not useful, simply reply the original answer.
Besides, briefly justify your answer in one or two sentences.
"""

class RefineResponseSynthesizer(BaseResponseSynthesizer):
  """The first chunk is given directly to the LLM as context information;
  In subsequent calls, give the previous result and the next chunk as context information"""
  
  def __init__(self, model: str = "gpt-3.5-turbo-1106",
               qa_prompt: str = DEFAULT_QA_PROMPT_TEMPLATE,
               refine_prompt: str = DEFAULT_REFINE_PROMPT_TEMPLATE,
               batch_size: int = 4) -> None:
    super().__init__(model)
    self.qa_prompt = qa_prompt
    self.refine_prompt = refine_prompt
    self.batch_size = batch_size
  
  def synthesize(self, question: str, context: [str]) -> (str, int):
    total_cost = 0
    
    prompt = self.qa_prompt.format(question = question, context = "\n".join(context[0 : self.batch_size]))
    response, cost = llm_call(model = self.model, user_prompt = prompt)
    total_cost += cost
    answer = response["choices"][0]["message"]["content"]
    # print(answer)
    
    for i in range(self.batch_size, len(context), self.batch_size):
      prompt = self.refine_prompt.format(question = question, answer = answer, context = "\n".join(context[i, i + self.batch_size]))
      # print(prompt)
      response, cost = llm_call(model = self.model, user_prompt = prompt)
      total_cost += cost
      answer = response["choices"][0]["message"]["content"]
      # print(answer)
    
    return answer, total_cost