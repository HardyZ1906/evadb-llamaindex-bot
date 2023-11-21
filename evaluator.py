from openai_utils import llm_call

DEFAULT_EVAL_PROMPT_TEMPLATE = """
You will be given some context information, a question
and a generated answer to the question.

Your job is to judge if the generated answer is supported
by the context. If the answer is "I don't know", you should
judge if the question really cannot be answered using the
context. Use the given context ONLY and not prior knowledge.

Reply in the first line "YES" or "NO" as your judgement.
Provide in a separate line a one-or-two-sentence reasoning
for your judgement.

Below is an example response:
-----------------------------------------------------------
YES
The answer can be concluded from the third paragraph.
-----------------------------------------------------------

Context:
-----------------------------------------------------------
{context}
-----------------------------------------------------------

Question:
-----------------------------------------------------------
{question}
-----------------------------------------------------------

Generated Answer:
-----------------------------------------------------------
{answer}
-----------------------------------------------------------
"""

class EvaluationResult:
  passing: bool
  score: str
  comments: str

class Evaluator:
  def __init__(self, model: str = "gpt-35-turbo", eval_template: str = DEFAULT_EVAL_PROMPT_TEMPLATE) -> None:
    self.model = model
    self.eval_template = eval_template
  
  def evaluate(self, context: str, question: str, answer: str) -> (EvaluationResult, int):
    prompt = self.eval_template.format(context = context, question = question, answer = answer)
    response, cost = llm_call(model = self.model, user_prompt = prompt)
    
    score, comments = response["choices"][0]["message"]["content"].strip("\n", 1)
    
    return EvaluationResult(
      passing = (score == "YES"),
      score = score,
      comments = comments
    ), cost
