from llama_index.core.tools import FunctionTool
import os

def code_reader_func(file_name):
    path=os.path.join("Data",file_name)
    try:
        with open(path,"r") as f:
            content=f.read()
            return {"file_content":content}
    except Exception as e:
            return {"error":str(e)}

code_reader =  FunctionTool.from_defaults(
    fn=code_reader_func,
    name="code_reader",
    description="This tool reads the content of a file and returns the results. Use this when you need to read content of a file."
)