import tiktoken

from pathlib import Path
import requests

import evadb

DEFAULT_CHUNK_SIZE = 1024

def split_text_into_chuncks(text: str, model: str = "gpt-3.5-turbo-1106", chunk_size: int = DEFAULT_CHUNK_SIZE) -> [str]:
  """Split `text` into chunks of size about (typically a bit above) `chunk_size` tokens of `model`'s"""
  enc = tiktoken.encoding_for_model(model_name = model)
  lines = text.split("\n")  # text is assumed to be line separated
  chunks = []
  curr_chunk = ""
  curr_chunk_size = 0
  for line in lines:
    curr_chunk += line
    curr_chunk_size += len(enc.encode(line))
    if curr_chunk_size >= chunk_size:
      chunks.append(curr_chunk)
      curr_chunk = ""
      curr_chunk_size = 0
  return chunks


def load_wiki_pages(page_titles: [str] = ["Toronto", "Boston", "Atlanta"],
                    doc: str = "cities") -> None:
  data_path = Path("data")
  if not data_path.exists():
    Path.mkdir(data_path)

  with open(data_path / f"{doc}.txt", "w") as f:
    pass

  for title in page_titles:
    response = requests.get(
      "https://en.wikipedia.org/w/api.php",
      params={
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        # 'exintro': True,
        "explaintext": True,
      },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]
    
    with open(data_path / f"{doc}.txt", "a") as f:
      f.write(wiki_text)
      f.write("\n")


def load_data_into_db(cursor: evadb.EvaDBCursor, doc: str = "cities") -> None:
  cursor.query(f"DROP TABLE IF EXISTS {doc};").df()
  cursor.query(f"LOAD DOCUMENT 'data/{doc}.txt' INTO {doc};").df()