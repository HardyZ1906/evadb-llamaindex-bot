import tiktoken

DEFAULT_CHUNK_SIZE = 1024

def split_text_into_chuncks(text: str, model: str = "gpt-4", chunk_size: int = DEFAULT_CHUNK_SIZE) -> [str]:
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