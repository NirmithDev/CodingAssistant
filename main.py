from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv

#load all variables and data from .env
load_dotenv()

llm = Ollama(model="gemma2:2b",request_timeout=60.0)
'''
result=llm.complete("Write a python program saying Hello World")
print(result)'''

parser = LlamaParse(result_type="markdown")
#atm it is just pdf supported will scale it for other formats
file_extractor = {".pdf": parser}
documents=SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
#locally hosted embedding model
embed_model=resolve_embed_model(embed_model="local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents,embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

result = query_engine.query("What are some of the routes of the api?")
print(result)