from openai_utils import llm_call
import evadb

DEFAULT_QUESTION_KEYWORD_EXTRACT_TEMPLATE = """
Consider the following question:
-----------------------------------------------------------
{question}
-----------------------------------------------------------

Please extract up to {max_keywords} keywords (can be phrases) from the question.
The keywords should capture the topic/main idea of the given question,
and will be used to look up the answer to the question.
Avoid extracting stop words.
Reply ONLY the semicolon-separated keywords, for example: "Atlanta;population"
"""

DEFAULT_TEXT_KEYWORD_EXTRACT_TEMPLATE = """
Consider the following text:
-----------------------------------------------------------
{text}
-----------------------------------------------------------

Please extract up to {max_keywords} keywords (can be phrases) from the text.
The keywords should capture the topic/main idea of the given text,
and will be used to categorize/look up the text.
Avoid extracting stop words.
Reply ONLY the semicolon-separated keywords, for example: "Atlanta;population"
"""

class KeywordTableIndex:
  """Extract key words from each text chunk and question;
  On retrieval, pick the K chunks having the most keywords in common with the question"""
  
  def __init__(self, cursor: evadb.EvaDBCursor, doc: str, max_keywords: int = 10,
               model: str = "gpt-4", init: bool = False,
               keyword_template = DEFAULT_TEXT_KEYWORD_EXTRACT_TEMPLATE) -> None:
    self.cursor = cursor
    self.doc = doc
    self.max_keywords = max_keywords
    self.model = model
    self.init_cost = 0
    self.keyword_table = {str:list}
    
    if init:
      docs = cursor.query(f"""
        SELECT _row_id, data FROM {self.doc};
      """).df()
      self.num_chunks = len(docs)

      prompt = keyword_template.format(text = self.doc, max_keywords = self.max_keywords)
      for _, row in docs.iterrows():
        response, cost = llm_call(model = self.model, user_prompt = prompt)
        self.init_cost += cost
        
        keywords = response["choices"][0]["message"]["content"]
        for kw in keywords.split(";"):
          if kw not in self.keyword_table.keys():
            self.keyword_table[kw] = [row["_row_id"]]
          else:
            self.keyword_table[kw].append(row["_row_id"])
          
        cursor.query(f"""
          DELETE FROM {self.doc} WHERE data = {row["data"]};
        """).df()
        cursor.query(f"""
          INSERT INTO {self.doc}(keywords, data) VALUES ({keywords}, {row["data"]});
        """).df()
    else:
      docs = cursor.query(f"""
        SELECT _row_id, keywords FROM {self.doc};
      """).df()
      self.num_chunks = len(docs)
      
      for _, row in docs.iterrows():
        for kw in row["keywords"].split(";"):
          if kw not in self.keyword_table.keys():
            self.keyword_table[kw] = [row["_row_id"]]
          else:
            self.keyword_table[kw].append(row["_row_id"])


  def retrieve(self, question: str, top_k: int = 4,
               keyword_template = DEFAULT_TEXT_KEYWORD_EXTRACT_TEMPLATE) -> ([str], int):
    total_cost = 0

    prompt = keyword_template.format(question = question, max_keywords = self.max_keywords)
    response, cost = llm_call(model = self.model, user_prompt = prompt)
    total_cost += cost
    keywords = set(response["choices"][0]["message"]["content"].split(";"))
    
    match_count = {int:int}
    for kw in keywords:
      for row_id in self.keyword_table[kw]:
        if row_id not in match_count.keys():
          match_count[row_id] = 1
        else:
          match_count[row_id] += 1
    rows = sorted(match_count.keys(), key = lambda x: match_count[x], reverse = True)[:top_k]
    
    chunks = []
    for row_id in rows:
      chunks.append(self.cursor.query(f"""
        SELECT data FROM {self.doc} WHERE _row_id = {row_id};
      """).df()["data"][0])

    return chunks, total_cost