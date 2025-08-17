# ==========================
# 🌐 Django & Project Settings
# ==========================
from seaborn import colors
from django.conf import settings
# from .models import Prompt
import matplotlib.pyplot as plt
import base64
from io import BytesIO
# import matplotlib.pyplot as plt


from io import StringIO
import pandas as pd

# ==========================
# 📦 Standard Library
# ==========================
import os
# import sys
# import uuid
import json
# import random
from datetime import datetime
from pprint import pprint

# ==========================
# 📦 Third-Party Core
# ==========================
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
# from typing_extensions import TypedDict

# ==========================
# 🧠 Google Generative AI
# ==========================
import google.generativeai as genai
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ==========================
# 🤖 LangChain Core & Community
# ==========================
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage
)
# from langchain_core.documents import Document
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
# from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ==========================
# 🔁 LangGraph Imports
# ==========================
from langgraph.graph import StateGraph, START, END, MessagesState
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph_checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver

# from langgraph.errors import NodeInterrupt


# --- Project-Specific Imports ---

#AJADI-2


# Load .env file
load_dotenv()
# from langgraph.checkpoint.sqlite import SqliteSaver # <--- Updated import

# Retrieve variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH", "default.pdf")


gemni = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm = gemni
model = llm # Consistent naming

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = InMemoryVectorStore(embeddings)

DB_URI = os.getenv("DB_URI")
db = SQLDatabase.from_uri(DB_URI)


class QueryOutput(BaseModel):
    query: str = Field(description="SQL query to run")
    QueryAnswer: str = Field(description="A clear, concise, empathetic, and polite response...")
    Chart_Type: str = Field(description="The type of chart to be generated")
    Chart_Data: str = Field(description="The data to be used for the chart")
    Chart_Title: str = Field(description="The title of the chart")
    Chart_X_Axis: str = Field(description="The x-axis of the chart")
    Chart_Y_Axis: str = Field(description="The y-axis of the chart")
    Chart_Legend: str = Field(description="The legend of the chart")
    Chart_Colors: str = Field(description="The colors of the chart")
    Chart_Size: str = Field(description="The size of the chart")
# Response schemas

# class Answer(BaseModel):
#     answerA: str
#     sentimentA: int
#     ticketA: list[str]
#     sourceA: list[str]


class Answer(BaseModel):
    answerA: str = Field(..., description="A clear, concise, empathetic, and polite response...")
    sentimentA: int = Field(..., description="An integer rating of the user's sentiment...")
    ticketA: list[str] = Field(..., description='A list of specific transaction or service channels...')
    sourceA: list[str] = Field(..., description='A list of specific sources...')


    
class Summary(BaseModel):
    """Conversation summary schema"""
    summaryS: str = Field(description="Summary of the entire conversation")
    sum_sentimentS: int = Field(description="Sentiment analysis of entire conversation")
    sum_ticketS: List[str] = Field(description="Channels with unresolved issues")
    sum_sourceS: List[str] = Field(description="All sources referenced in conversation")


class State(MessagesState):
    """State management for conversation flow"""
    question: str
    pdf_content: str
    web_content: str
    query_answerT: str
    answer: str
    sentiment: int
    ticket: List[str]
    source: List[str]
    attached_content:str
    summary : str
    sum_sentiment: int
    sum_ticket: List[str]
    sum_source: List[str]
    answerY: Answer
    metadatas: Dict[str, Any] = Field(default_factory=dict)
    summaryY: Summary


def get_time_based_greeting():
    """Return an appropriate greeting based on the current time."""
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 17:
        return "Good afternoon"
    elif 17 <= current_hour < 22:
        return "Good evening"
    else:
        return "Good night"

def get_current_datetime():
    """Returns the current date and time in YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def safe_json(data):
    """Ensures safe JSON serialization to prevent errors."""
    try:
        return json.dumps(data)
    except (TypeError, ValueError):
        return json.dumps({})  # Returns an empty JSON object if serialization fails


dialect = db.dialect # Corrected from tuple
top_k = 10
table_info = db.get_table_info()
current_time = get_current_datetime()


sql_prompt = f"""
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Kindly note the current time: {current_time},
Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""
sys_msg = SystemMessage(content=sql_prompt)
model_with_structure = llm.with_structured_output(QueryOutput) # Use llm here

# Pass only the relevant part of the state, not the whole state for invoke

user_input="Give me the top 5 customers by sales and plot a bar chart?"
result = model_with_structure.invoke([sys_msg] + [HumanMessage(content=user_input)])
execute_query_tool = QuerySQLDatabaseTool(db=db)

if hasattr(result, "query"):
    result_query=result.query
    resultT = execute_query_tool.invoke(result_query)
    print ("AJADI Quesry ",resultT)
else:
    print("⚠️ Unexpected Quesry result format. Raw output:", result)
    result_query = "[Query not available]"
    resultT = "[No result]"

# # Alternative approach using SQLAlchemy and pandas:
from sqlalchemy import create_engine

# Create a database engine
engine = create_engine(DB_URI)
query = result_query
df = pd.read_sql_query(query, con=engine)
data_str = df.to_csv(index=False)

column_names = df.columns.tolist()
print("AJADI Column Names :",column_names)
prompt=user_input
# print("AJADI Pandas Preview Second Agoba :",df)

# Convert string data into a DataFrame
# df = pd.read_csv(StringIO(data_str))

# Request insights from Gemini
response = model.invoke(f"Analyze this data and provide insights based on this prompt: {prompt}\n\nData:\n{data_str}")
print(df.head())
print(df.dtypes)




# import numpy as np

# 1. Detect numeric columns
numeric_cols = df.select_dtypes(include='number').columns.tolist()

# 2. Detect categorical columns (object, string, or low-cardinality numeric)
categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

# If no obvious categorical columns, look for low-cardinality numeric columns
if not categorical_cols:
    # Heuristic: treat numeric columns with few unique values as categorical
    n_rows = len(df)
    for col in numeric_cols:
        n_unique = df[col].nunique(dropna=True)
        if n_unique < 20 or n_unique < n_rows * 0.5:
            categorical_cols.append(col)

# Remove columns that are both in numeric and categorical (if any)
categorical_cols = [col for col in categorical_cols if col not in numeric_cols]

# Now, select columns for plotting
if numeric_cols and categorical_cols:
    y_column = numeric_cols[0]
    x_column = categorical_cols[0]
elif len(numeric_cols) >= 2:
    # If all columns are numeric, use the one with the most unique values as x, the other as y
    unique_counts = {col: df[col].nunique() for col in numeric_cols}
    x_column = max(unique_counts, key=unique_counts.get)
    y_column = min(unique_counts, key=unique_counts.get)
else:
    raise ValueError("Could not automatically determine numeric and categorical columns.")

print(f"Using x_column: {x_column}, y_column: {y_column}")





# # Identify columns
# numeric_cols = df.select_dtypes(include='number').columns.tolist()
# categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

# if numeric_cols and categorical_cols:
#     y_column = numeric_cols[0]
#     x_column = categorical_cols[0]
# else:
#     raise ValueError("Could not automatically determine numeric and categorical columns.")


# # Check if visualization is needed
# viz_needed = any(word in prompt.lower() for word in ['graph', 'chart', 'plot', 'visual'])


# matched_word = next((word for word in ['graph', 'chart', 'plot', 'visual'] if word in prompt.lower()), None)
# if matched_word:
#     print(f"Found: {matched_word}")
# else:
#     print("No visualization word found.")




# if viz_needed:
#    # Plot
#     plt.figure(figsize=(8, 6))
#     # plt.bar(df[x_column].astype(str), df[y_column], color='skyblue',)
#     plt.bar(df[x_column].astype(str), df[y_column], color=plt.cm.Paired.colors )
#     plt.xlabel(x_column.capitalize())
#     plt.ylabel(y_column.capitalize())
#     plt.title(f'{y_column.capitalize()} by {x_column.capitalize()}')
#     plt.tight_layout()
    

    # plt.pie(
    #     df[y_column],
    #     labels=df[x_column],
    #     autopct='%1.1f%%',
    #     startangle=140,
    #     colors=plt.cm.Paired.colors  # adds a splash of style
    # )
    # plt.title('Total Sales Distribution by Date')
    # plt.axis('equal')  # Ensures perfect circle
    # Save chart to buffer and encode as base64


# Check if any visualization keyword is in the prompt


# chart_map = {
#     'bar': 'bar',
#     'bar chart': 'bar',
#     'column chart': 'bar',
#     'line': 'line',
#     'scatter': 'scatter',
#     'pie': 'pie',
#     'pie chart': 'pie',
#     'donut chart': 'pie',
#     'histogram': 'histogram',
#     'boxplot': 'box',
#     'box plot': 'box',
#     'heatmap': 'heatmap',
#     'area': 'area',
#     'area chart': 'area',
#     'bubble': 'bubble'
# }
# prompt = prompt.lower()  # Normalize
# matched_chart = next((chart_map[word] for word in chart_map if word in prompt), None)
# def recommend_chart(df, matched_chart=None):
#     num_cols = len(df.columns)

#     if matched_chart:
#         return matched_chart
    
#     # Fallback based on column count or types
#     if num_cols == 1:
#         return 'histogram'
#     elif num_cols == 2:
#         col_types = df.dtypes
#         if 'object' in col_types.values and 'float' in col_types.values or 'int' in col_types.values:
#             return 'bar'
#         else:
#             return 'scatter'
#     else:
#         return 'line'




# viz_keywords = ['graph', 'chart', 'plot', 'visual', 'pie', 'bar chart', 'histogram', 'line']
# viz_needed = any(word in prompt.lower() for word in viz_keywords)

# # Find the first matched keyword
# matched_word = next((word for word in viz_keywords if word in prompt.lower()), None)

# if viz_needed:
#     plt.figure(figsize=(8, 6))

#     # Pick the right plot based on keyword and column count
#     if matched_word == "histogram" and len(df.columns) == 1:
#         plt.hist(df[df.columns[0]], color='skyblue')  # Assuming single numeric column

#     elif matched_word == "pie" and len(df.columns) == 2:
#         plt.pie(df[y_column], labels=df[x_column].astype(str), colors=plt.cm.Paired.colors,
#                 autopct='%1.1f%%', startangle=90)
#         plt.axis('equal')  # Optional for pie charts

#     elif matched_word in ["bar chart", "bar"] and len(df.columns) >= 2:
#         plt.bar(df[x_column].astype(str), df[y_column], color=plt.cm.Paired.colors)

#     elif matched_word == "line" and len(df.columns) >= 2:
#         plt.plot(df[x_column], df[y_column], color='dodgerblue', marker='o')

#     else:
#         # Default to bar chart
#         plt.bar(df[x_column].astype(str), df[y_column], color=plt.cm.Paired.colors)

#     plt.xlabel(x_column.capitalize())
#     plt.ylabel(y_column.capitalize())
#     plt.title(f"{y_column.capitalize()} by {x_column.capitalize()}")
#     plt.tight_layout()
#     plt.show()
    

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

# Chart mapping
chart_map = {
    'bar': 'bar',
    'bar chart': 'bar',
    'column chart': 'bar',
    'line': 'line',
    'scatter': 'scatter',
    'pie': 'pie',
    'pie chart': 'pie',
    'donut chart': 'pie',
    'histogram': 'histogram',
    'boxplot': 'box',
    'box plot': 'box',
    'heatmap': 'heatmap',
    'area': 'area',
    'area chart': 'area',
    'bubble': 'bubble'
}

# Prompt detection
prompt = prompt.lower()
matched_chart = next((chart_map[word] for word in chart_map if word in prompt), None)
viz_needed = matched_chart is not None

def recommend_chart(df, matched_chart=None):
    if matched_chart:
        return matched_chart

    num_cols = len(df.columns)
    if num_cols == 1:
        return 'histogram'
    elif num_cols == 2:
        col_types = df.dtypes
        if any(t in ['float64', 'int64'] for t in col_types.values):
            return 'bar'
        else:
            return 'scatter'
    else:
        return 'line'

# Comma formatter
comma_formatter = FuncFormatter(lambda x, _: f"{int(x):,}")

if viz_needed:
    chart_type = recommend_chart(df, matched_chart)
    plt.figure(figsize=(8, 6))

    if chart_type == 'histogram' and len(df.columns) == 1:
        plt.hist(df[df.columns[0]], color='skyblue', edgecolor='black')
        plt.gca().yaxis.set_major_formatter(comma_formatter)

    elif chart_type == 'pie' and len(df.columns) == 2:
        plt.pie(df[y_column], labels=df[x_column].astype(str),
                colors=plt.cm.Paired.colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')

    elif chart_type == 'line':
        plt.plot(df[x_column], df[y_column], color='dodgerblue', marker='o')
        plt.gca().yaxis.set_major_formatter(comma_formatter)

    elif chart_type == 'scatter':
        plt.scatter(df[x_column], df[y_column], color='green')
        plt.gca().yaxis.set_major_formatter(comma_formatter)

    elif chart_type == 'box':
        plt.boxplot(df.select_dtypes(include=['float', 'int']).values)
        plt.gca().yaxis.set_major_formatter(comma_formatter)

    else:  # fallback to bar chart
        plt.bar(df[x_column].astype(str), df[y_column], color=plt.cm.Paired.colors)
        plt.gca().yaxis.set_major_formatter(comma_formatter)

    if chart_type != 'pie':
        plt.xlabel(x_column.capitalize())
        plt.ylabel(y_column.capitalize())
        plt.title(f"{chart_type.capitalize()} of {y_column} by {x_column}")

    plt.tight_layout()
    # plt.show()
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    image_base64 = base64.b64encode(image_png).decode('utf-8')
    plt.close()  # Close the plot to free memory
    # print("AJADI Image Base64 :",image_base64)

    ay= (response.text, image_base64)
else:
  ay= (response.text, None)

# Main execution block
if __name__ == "__main__":
    # Example usage
    result = ay
    # print("Result:", result)
    
    # Example of how to display the image in different contexts
    if result and len(result) > 1 and result[1]:
        text_response, image_base64 = result
        
        print("\n" + "="*50)
        print("WAYS TO DISPLAY THE BASE64 IMAGE")
        print("="*50)
        
        # Method 1: HTML img tag (for Django templates)
        html_img_tag = f'<img src="data:image/png;base64,{image_base64}" alt="Generated Chart" class="img-fluid">'
        print("\n1. HTML IMG TAG (for Django templates):")
        # print(html_img_tag)
        
        # Method 2: JavaScript/AJAX response
        js_response = {
            'text': text_response,
            'image_base64': image_base64,
            'html_image': html_img_tag
        }
        print("\n2. JavaScript/AJAX Response:")
        # print(js_response)
        
        # Method 3: Django context for template
        django_context = {
            'chart_text': text_response,
            'chart_image': image_base64,
            'chart_html': html_img_tag
        }
        print("\n3. Django Context for Template:")
        # print(django_context)
        
        # Method 4: Save to file
        try:
            import os
            media_dir = 'media/charts/'
            os.makedirs(media_dir, exist_ok=True)
            
            image_path = os.path.join(media_dir, 'generated_chart.png')
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_base64))
            # print(f"\n4. Saved to file: {image_path}")
        except Exception as e:
            print(f"\n4. Error saving file: {e}")
        
        # Method 5: Display in Jupyter/IPython (if available)
        try:
            from IPython.display import Image, display
            print("\n5. For Jupyter/IPython display:")
            # print("display(Image(base64.b64decode(image_base64)))")
        except ImportError:
            print("\n5. IPython not available for display")
        
        print("\n" + "="*50)


#WHATSAPP

# def generate_chart(prompt="Plot the average monthly sales in the last 5 months as a bar chart.", data_str=None):
#     """
#     Generate a chart based on the given prompt and data.
    
#     Args:
#         prompt (str): The prompt for chart generation
#         data_str (str): CSV data string. If None, uses default sample data
    
#     Returns:
#         tuple: (response_text, image_base64) or (response_text, None) if no visualization
#     """
#     if data_str is None:
#         # Sample prompt and embedded CSV data string
#         data_str = data_str
#     data_str = data_str
#     # Convert string data into a DataFrame
#     df = pd.read_csv(StringIO(data_str))

#     # Request insights from Gemini
#     response = model.invoke(f"Analyze this data and provide insights based on this prompt: {prompt}\n\nData:\n{data_str}")


#     column_names = df.columns.tolist()
#     print("AJADI Column Names :",column_names)
#     # prompt=user_input
#     # print("AJADI Pandas Preview Second Agoba :",df)
#     # data_str = df.to_csv(index=False)


# # Convert string data into a DataFrame
# # df = pd.read_csv(StringIO(data_str))

# # Request insights from Gemini
#     y_column=column_names[0]
#     x_column=column_names[1]



#     # Check if visualization is needed
#     viz_needed = any(word in prompt.lower() for word in ['graph', 'chart', 'plot', 'visual'])

#     if viz_needed:
#         plt.figure(figsize=(8, 8))
#         plt.pie(
#             df[y_column],
#             labels=df[x_column],
#             autopct='%1.1f%%',
#             startangle=140,
#             colors=plt.cm.Paired.colors  # adds a splash of style
#         )
#         plt.title('Total Sales Distribution by Date')
#         plt.axis('equal')  # Ensures perfect circle
#         # Save chart to buffer and encode as base64
#         buffer = BytesIO()
#         plt.tight_layout()
#         plt.savefig(buffer, format='png')
#         buffer.seek(0)
#         image_png = buffer.getvalue()
#         buffer.close()
#         image_base64 = base64.b64encode(image_png).decode('utf-8')
#         plt.close()  # Close the plot to free memory

#         return (response.text, image_base64)
#     else:
#         return (response.text, None)

# # Main execution block
# if __name__ == "__main__":
#     # Example usage
#     result = generate_chart()
#     print("Result:", result)
    
#     # Example of how to display the image in different contexts
#     if result and len(result) > 1 and result[1]:
#         text_response, image_base64 = result
        
#         print("\n" + "="*50)
#         print("WAYS TO DISPLAY THE BASE64 IMAGE")
#         print("="*50)
        
#         # Method 1: HTML img tag (for Django templates)
#         html_img_tag = f'<img src="data:image/png;base64,{image_base64}" alt="Generated Chart" class="img-fluid">'
#         print("\n1. HTML IMG TAG (for Django templates):")
#         print(html_img_tag)
        
#         # Method 2: JavaScript/AJAX response
#         js_response = {
#             'text': text_response,
#             'image_base64': image_base64,
#             'html_image': html_img_tag
#         }
#         print("\n2. JavaScript/AJAX Response:")
#         print(js_response)
        
#         # Method 3: Django context for template
#         django_context = {
#             'chart_text': text_response,
#             'chart_image': image_base64,
#             'chart_html': html_img_tag
#         }
#         print("\n3. Django Context for Template:")
#         print(django_context)
        
#         # Method 4: Save to file
#         try:
#             import os
#             media_dir = 'media/charts/'
#             os.makedirs(media_dir, exist_ok=True)
            
#             image_path = os.path.join(media_dir, 'generated_chart.png')
#             with open(image_path, 'wb') as f:
#                 f.write(base64.b64decode(image_base64))
#             print(f"\n4. Saved to file: {image_path}")
#         except Exception as e:
#             print(f"\n4. Error saving file: {e}")
        
#         # Method 5: Display in Jupyter/IPython (if available)
#         try:
#             from IPython.display import Image, display
#             print("\n5. For Jupyter/IPython display:")
#             print("display(Image(base64.b64decode(image_base64)))")
#         except ImportError:
#             print("\n5. IPython not available for display")
        
#         print("\n" + "="*50)



