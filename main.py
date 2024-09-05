from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
from prompts import context
from code_reader import code_reader

#load all variables and data from .env
load_dotenv()

llm = Ollama(model="gemma2:2b")
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

tools= [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="this gives documentation about code for an API. Use this for reading documentation for an API"
        ),
    ),
    code_reader,
]

code_llm=Ollama(model="deepseek-coder:1.3b")
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

while (prompt:=input("enter your prompt: (q to quit)"))!="q":
    response=agent.query(prompt)
    print(response)