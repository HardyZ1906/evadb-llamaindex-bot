from openai_utils import llm_call
import evadb
from retriever.base import BaseRetriever

DEFAULT_QUESTION_KEYWORD_EXTRACT_TEMPLATE = """
Consider the following question:
-----------------------------------------------------------
{question}
-----------------------------------------------------------

Please extract up to {max_keywords} keywords from the question.
The keywords should capture the topic/main idea of the given question,
and will be used to look up the answer to the question.
Avoid extract stop words.
Reply ONLY the keywords, one in a separate line, and nothing else. For example:
-----------------------------------------------------------
Atlanta
population
-----------------------------------------------------------

Keywords:
"""

DEFAULT_QUESTION_KEYPHRASE_EXTRACT_TEMPLATE = """
Consider the following question:
-----------------------------------------------------------
{question}
-----------------------------------------------------------

Please extract up to {max_keywords} concise keyphrases from the question.
The keyphrases should capture the topic/main idea of the given question,
and will be used to look up the answer to the question.
Avoid extracting stop words.
Reply ONLY the keyphrases, one in a separate line, and nothing else. For example:
-----------------------------------------------------------
Atlanta
capital city of Georgia
-----------------------------------------------------------

Keywords:
"""

DEFAULT_TEXT_KEYWORD_EXTRACT_TEMPLATE = """
Consider the following text:
-----------------------------------------------------------
{text}
-----------------------------------------------------------

Please use up to {max_keywords} keywords to summarize the text.
The keywords should capture the topic/main idea of the given text,
and will be used to categorize/look up the text.
Avoid extracting stop words.
Reply ONLY the keywords, one in a separate line, and nothing else. For example:
-----------------------------------------------------------
Atlanta
population
-----------------------------------------------------------

Keywords:
"""

DEFAULT_TEXT_KEYPHRASE_EXTRACT_TEMPLATE = """
Consider the following text:
-----------------------------------------------------------
{text}
-----------------------------------------------------------

Please use up to {max_keywords} concise keyphrases to summarize the text.
The keyphrases should capture the topic/main idea of the given text,
and will be used to categorize/look up the text.
Reply ONLY the keyphrases, one in a separate line, and nothing else. For example:
-----------------------------------------------------------
Atlanta
capital city of Georgia
-----------------------------------------------------------

Keywords:
"""

class KeywordTableIndexRetriever(BaseRetriever):
  """Extract key words from each text chunk and question;
  On retrieval, pick the K chunks having the most keyword matches;
  Keywords can be matched using either exact match or semantic search"""
  
  def __init__(self, cursor: evadb.EvaDBCursor, doc: str, max_keywords: int = 20,
               model: str = "gpt-3.5-turbo-1106", top_k: int = 8, new: bool = False, exact_match: bool = True,
               text_template = None,
               question_template = None) -> None:
    super().__init__(cursor, doc)
    self.max_keywords = max_keywords
    self.model = model
    self.init_cost = 0
    self.keyword_table = {str:list}
    self.top_k = top_k
    self.exact_match = exact_match
    self.text_template = text_template or \
      DEFAULT_TEXT_KEYWORD_EXTRACT_TEMPLATE \
      if self.exact_match else \
      DEFAULT_QUESTION_KEYPHRASE_EXTRACT_TEMPLATE
    self.question_template = question_template or \
      DEFAULT_QUESTION_KEYWORD_EXTRACT_TEMPLATE \
      if self.exact_match else \
      DEFAULT_QUESTION_KEYPHRASE_EXTRACT_TEMPLATE
    
    if new:
      self.cursor.query(f"""
        DROP TABLE IF EXISTS {self.doc}_keywords;
      """).df()
      self.cursor.query(f"""
        CREATE TABLE {self.doc}_keywords(chunk_id INTEGER, keyword TEXT(50));
      """).df()
      
      docs = self.cursor.query(f"""
        SELECT chunk_id, data FROM {self.doc};
      """).df()
      # print(f"{len(docs)} chunks")

      for _, row in docs.iterrows():
        prompt = self.text_template.format(text = row[f"{self.doc}.data"], max_keywords = self.max_keywords)
        # print(f"--------------------\n{prompt}")
        response, cost = llm_call(model = self.model, user_prompt = prompt)
        self.init_cost += cost
        
        keywords = response["choices"][0]["message"]["content"].lower()
        # print(f"--------------------\n{keywords}")

        for kw in keywords.split("\n"):
          self.cursor.query(f"""
            INSERT INTO {self.doc}_keywords(chunk_id, keyword) VALUES ({row[f"{self.doc}.chunk_id"]}, "{kw.strip()}");
          """).df()
      
      cursor.query(f"""
        SELECT * FROM {self.doc}_keywords;
      """).df()[[f"{self.doc}_keywords.chunk_id", f"{self.doc}_keywords.keyword"]].to_csv(f"./{self.doc}_keywords.csv", index = False)
      
      if not self.exact_match:
        self.cursor.query("""
          CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
          IMPL './sentence_feature_extractor.py';
        """).df()
        self.cursor.query(f"""
          DROP INDEX IF EXISTS {self.doc}_keywords_index;
        """).df()
        self.cursor.query(f"""
          CREATE INDEX {self.doc}_keywords_index
          ON {self.doc}_keywords(SentenceFeatureExtractor(keyword))
          USING FAISS;
        """).df()
    
    if self.exact_match:
      keywords = self.cursor.query(f"""
        SELECT chunk_id, keyword FROM {self.doc}_keywords;
      """).df()
      for _, row in keywords.iterrows():
        kw = row[f"{self.doc}_keywords.keyword"].lower()
        chunk_id = row[f"{self.doc}_keywords.chunk_id"]
        if kw not in self.keyword_table.keys():
          self.keyword_table[kw] = [chunk_id]
        else:
          self.keyword_table[kw].append(chunk_id)


  def retrieve(self, question: str) -> ([str], int):
    total_cost = 0

    prompt = self.question_template.format(question = question, max_keywords = self.max_keywords)
    # print(prompt)
    response, cost = llm_call(model = self.model, user_prompt = prompt)
    total_cost += cost
    ans = response["choices"][0]["message"]["content"]
    # print(ans)
    keywords = set(ans.split("\n"))
    
    match_count = {}
    if self.exact_match:
      for kw in keywords:
        if kw.lower() not in self.keyword_table.keys():
          continue
        for chunk_id in self.keyword_table[kw.lower()]:
          if chunk_id not in match_count.keys():
            match_count[chunk_id] = 1
          else:
            match_count[chunk_id] += 1
    else:
      for kw in keywords:
        matches = self.cursor.query(f"""
          SELECT * FROM {self.doc}_keywords
          ORDER BY
            Similarity(
              SentenceFeatureExtractor(f"{kw}"),
              SentenceFeatureExtractor(keyword)
            )
          LIMIT {self.top_k}
        """).df()
        for _, row in matches.iterrows():
          chunk_id = row[f"{self.doc}_keywords.chunk_id"]
          # print(f"""matched with {chunk_id}: {row[f"{self.doc}_keywords.keyword"]}""")
          if chunk_id not in match_count.keys():
            match_count[chunk_id] = 1
          else:
            match_count[chunk_id] += 1
    
    chunk_ids = sorted(match_count.keys(), key = lambda x: match_count[x], reverse = True)[:self.top_k]
    chunks = []
    for chunk_id in chunk_ids:
      chunks.append(self.cursor.query(f"""
        SELECT data FROM {self.doc} WHERE chunk_id = {chunk_id};
      """).df()[f"{self.doc}.data"][0])
    # print("\n\n\n".join(chunks))

    return chunks, total_cost