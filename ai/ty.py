from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

start_time = datetime.now()
print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

template = """Question: {question}
Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)

# Explicitly set base_url to avoid WinError 10049
# model = OllamaLLM(model="llama3", base_url="http://127.0.0.1:11434")
model = OllamaLLM(model="gpt-oss:20b", base_url="http://127.0.0.1:11434")

chain = prompt | model

response = chain.invoke({"question": "What is LangChain?"})
print(response)

start_time = datetime.now()
print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
