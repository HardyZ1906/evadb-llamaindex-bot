import os
import evadb

from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext, SimpleWebPageReader, load_index_from_storage
from llama_index.prompts import PromptTemplate


standard_qa_template_str = (
  "We provide you with some context information and a question. Please answer the question with a code snippet. Do not repeat this prompt.\n"

  "Here is the context information:\n"
  "----------------------------------------------------------------\n"
  "{context_str}"
  "----------------------------------------------------------------\n"

  "Given this information, please answer the question: {query_str}\n"
)

succinct_qa_template_str = (
  "We provide you with some context information and a question. Please answer the question with a code snippet. When you write codes, please omit parts that are lengthy but straight-forward or marginally relevant, like environmental setup - replace those with a single-line comment or pseudo-codes. Also, you can assume that no errors or exceptions would occur, so error handling is unnecessary. In short, only give the most important and pertinent code. Do not repeat this prompt.\n"

  "Here is the context information:\n"
  "----------------------------------------------------------------\n"
  "{context_str}\n"
  "----------------------------------------------------------------\n"

  "Given this information, please answer the question: {query_str}\n"
)


def build_index() -> GPTVectorStoreIndex:
  documents = SimpleWebPageReader(html_to_text=True).load_data(
    [
      "https://libvirt.org/html/libvirt-libvirt-common.html",
      # "https://libvirt.org/html/libvirt-libvirt-domain-checkpoint.html",
      # "https://libvirt.org/html/libvirt-libvirt-domain-snapshot.html",
      "https://libvirt.org/html/libvirt-libvirt-domain.html",
      # "https://libvirt.org/html/libvirt-libvirt-event.html",
      "https://libvirt.org/html/libvirt-libvirt-host.html",
      # "https://libvirt.org/html/libvirt-libvirt-interface.html",
      # "https://libvirt.org/html/libvirt-libvirt-network.html",
      # "https://libvirt.org/html/libvirt-libvirt-nodedev.html",
      # "https://libvirt.org/html/libvirt-libvirt-nwfilter.html",
      # "https://libvirt.org/html/libvirt-libvirt-secret.html",
      # "https://libvirt.org/html/libvirt-libvirt-storage.html",
      # "https://libvirt.org/html/libvirt-libvirt-stream.html"
    ]
  )
  
  service_context = ServiceContext.from_defaults(chunk_size = 512)
  index = GPTVectorStoreIndex.from_documents(documents, service_context = service_context, show_progress = True)
  index.set_index_id("index_libvirt")
  index.storage_context.persist("./llama_index")
  
  return index


def load_index() -> GPTVectorStoreIndex:
  storage_context = StorageContext.from_defaults(persist_dir = "./llama_index")
  return load_index_from_storage(storage_context = storage_context, index_id = "index_libvirt")


def get_user_input() -> dict:
  print("Welcome! This is a `libvirt` programming helper bot based on EvaDB and Llamaindex.\nWe can answer your questions regarding programming with `libvirt` using example code snippets.")
  
  query_str = input("Please enter your question: ")
  succinct = input("Do you want the code snippet in the answer to be succinct (i.e. containing only the most informative code)? (y/n)").lower() in ['y', 'yes']
  
  return {"query": query_str, "succinct": succinct}


if __name__ == "__main__":
  # cursor = evadb.connect().cursor()
  # drop_table(cursor)
  # load_table(cursor)
  # print(cursor.query("SHOW TABLES;").df())

  OPENAI_API_KEY = "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"
  os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
  
  index = load_index() if len(os.listdir("./llama_index")) > 0 else build_index()

  user_input = get_user_input()
  qa_template = PromptTemplate(succinct_qa_template_str) if user_input["succinct"] else PromptTemplate(standard_qa_template_str)
  query_engine = index.as_query_engine(text_qa_template = qa_template)
  result = query_engine.query(user_input["query"])

  print(result)
  