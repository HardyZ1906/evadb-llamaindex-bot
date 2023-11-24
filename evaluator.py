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
  
  def __init__(self, passing: bool, score: str, comments: str) -> None:
    self.passing = passing
    self.score = score
    self.comments = comments

class Evaluator:
  def __init__(self, model: str = "gpt-3.5-turbo-1106", eval_template: str = DEFAULT_EVAL_PROMPT_TEMPLATE) -> None:
    self.model = model
    self.eval_template = eval_template
  
  def evaluate(self, context: str, question: str, answer: str) -> (EvaluationResult, int):
    prompt = self.eval_template.format(context = context, question = question, answer = answer)
    # print(prompt)
    response, cost = llm_call(model = self.model, user_prompt = prompt)
    
    ans = response["choices"][0]["message"]["content"]
    # print(ans)
    score, comments = ans.split("\n", 1)
    
    return EvaluationResult(
      passing = (score == "YES"),
      score = score,
      comments = comments
    ), cost
