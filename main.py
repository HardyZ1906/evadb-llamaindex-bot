from query_engine.simple import SimpleQueryEngine
from query_engine.retry import RetryQueryEngine

from retriever.keyword_table_index import KeywordTableIndexRetriever
from retriever.summary_index import SummaryIndexRetriever
from retriever.vector_store_index import VectorStoreIndexRetriever

from response_synthesizer.compact import CompactResponseSynthesizer
from response_synthesizer.refine import RefineResponseSynthesizer
from response_synthesizer.tree_summarize import TreeSummarizeResponseSynthesizer

from evaluator import Evaluator

from utils import load_wiki_pages, load_data_into_db

import os
import openai
import evadb

from getpass import getpass

if __name__ == "__main__":
  print("⏳ Connecting to EvaDB...")
  cursor = evadb.connect().cursor()
  print("✅ Connected to EvaDB!")
  
  first = False
  doc = "cities"
  
  if os.getenv("OPENAI_API_KEY") is None:
    api_key = getpass("Please provide your OpenAI API key (will be hidden): ")
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key
  
  if first:
    print("Loading data...")
    load_wiki_pages(doc = doc)
    load_data_into_db(cursor)
    print("Data loaded!")
    
    print("Building keyword table...")
    keyword_table_index_retriever = KeywordTableIndexRetriever(cursor, doc, new = True)
    print("Keyword table built!")
  
  # retriever = SummaryIndexRetriever(cursor, doc)
  retriever = VectorStoreIndexRetriever(cursor, doc)
  # retriever = KeywordTableIndexRetriever(cursor, doc)
  
  response_synthesizer = CompactResponseSynthesizer()
  # response_synthesizer = RefineResponseSynthesizer()
  # response_synthesizer = TreeSummarizeResponseSynthesizer()
  
  # evaluator = Evaluator()
  
  query_engine = SimpleQueryEngine(retriever, response_synthesizer)
  # query_engine = RetryQueryEngine(retriever, response_synthesizer, evaluator)
  
  while True:
    question = input("Please enter your question: ")
    answer, cost = query_engine.query(question)
    print(f"answer: {answer}")
    print(f"cost: {cost}")
    if input("Do you have any other questions? (y/n)\n").lower() not in ["y", "yes"]:
      break
  
  # query_str = ""
  # while True:
  #   query_str += input()
  #   if query_str.endswith(";"):
  #     try:
  #       print(cursor.query(query_str).df())
  #     except Exception as e:
  #       print(e)
  #     query_str = ""