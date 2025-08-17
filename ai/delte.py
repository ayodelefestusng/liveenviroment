

model = llm # Consistent naming

# Initialize Embeddings and Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", transport="rest")
global_vector_store = None

def initialize_vector_store():
    """Initializes the vector store by loading and splitting the PDF document."""
    global global_vector_store
    if global_vector_store is None:
        global_vector_store = InMemoryVectorStore(embedding=embeddings)
        
        # Use the PDF_PATH from environment variables or default
        # file_path = PDF_PATH # This should be a full path or handled by Django settings
        file_path = os.path.join(settings.MEDIA_ROOT, 'pdfs', 'ATB Bank Nigeria Groq v2.pdf')
        
        # For local testing without Django settings, you might hardcode or derive it:
        # file_path = r"C:\Users\Pro\Desktop\Ayodele\25062025\myproject\media\pdfs\ATB Bank Nigeria Groq v2.pdf"

        if file_path and os.path.exists(file_path):
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
                all_splits = text_splitter.split_documents(docs)
                global_vector_store.add_documents(documents=all_splits)
                print("PDF document loaded and processed successfully.")
            except Exception as e:
                print(f"Error loading PDF: {e}. PDF retrieval tool will not work.")
        else:
            print(f"Warning: PDF file not found at {file_path}. PDF retrieval tool will not work.")

# Call the initialization function
initialize_vector_store()

# Initialize SQLDatabase with the specified file path
# DB_FILE_PATH = r"C:\Users\Pro\Desktop\Ayodele\25062025\myproject\db.sqlite3"
DB_FILE_PATH = r"C:\Users\Pro\Desktop\PROJECT\Live\myproject\db.sqlite3"
DB_URI = f"sqlite:///{DB_FILE_PATH}" 
# DB_URI = os.getenv("DB_URI")
db = None
try:
    db = SQLDatabase.from_uri(DB_URI)
    print(f"SQLDatabase connected to {DB_URI} successfully.")
except Exception as e:
    print(f"Error connecting to SQLDatabase at {DB_URI}: {e}. SQL query tool will not be available.")


# ==========================
# 📝 Pydantic Schemas
# ==========================
# class Answer(BaseModel):
#     answerA: str = Field(..., description="A clear, concise, empathetic, and polite response...")
#     sentimentA: int = Field(..., description="An integer rating of the user's sentiment...")
#     ticketA: list[str] = Field(..., description='A list of specific transaction or service channels...')
#     sourceA: list[str] = Field(..., description='A list of specific sources...')

class Answer(BaseModel):
    """The final, structured answer for the user."""
    answer: str = Field(description="Polite, empathetic, and direct response to the user's query.")
    sentiment: int = Field(description="User's sentiment score from -2 (very negative) to +2 (very positive).")
    ticket: List[str] = Field(description="Relevant service channels for unresolved issues (e.g., 'POS', 'ATM').")
    source: List[str] = Field(description="Sources used to generate the answer (e.g., 'PDF Content', 'Web Search').")

class Summary(BaseModel):
    """Conversation summary schema."""
    summary: str = Field(description="A concise summary of the entire conversation.")
    sentiment: int = Field(description="Overall sentiment of the conversation from -2 to +2.")
    unresolved_tickets: List[str] = Field(description="A list of channels with unresolved issues.")
    all_sources: List[str] = Field(description="All unique sources referenced throughout the conversation.")

class PDFRetrievalInput(BaseModel):
    """Input schema for the pdf_retrieval_tool."""
    query: str = Field(description="The user's query to search for within the PDF document.")

class WebSearchInput(BaseModel):
    """Input schema for the tavily_search_tool."""
    query: str = Field(description="A concise search query for the web.")

class SQLQueryInput(BaseModel):
    """Input schema for the sql_query_tool."""
    query: str = Field(description="The natural language question to be converted into a SQL query.")

# ==========================
# 📊 State Management (Simplified and Centralized)
# ==========================

class State(MessagesState):
    """Manages the conversation state. Uses Pydantic models for structured data."""
    # Tool outputs
    pdf_content: Optional[str] = None
    web_content: Optional[str] = None
    sql_result: Optional[str] = None
    attached_content: Optional[str] = None

    # Final structured outputs
    final_answer: Optional[Answer] = None
    conversation_summary: Optional[Summary] = None

    # For final logging
    metadatas: Optional[Dict[str, Any]] = None

# ==========================
# 🛠️ Tools
# ==========================

def retrieve_from_pdf(query: str) -> dict:
    """Performs a document query using the initialized vector store."""
    if global_vector_store:
        results = global_vector_store.similarity_search(query, k=3)
        content = "\n\n".join([doc.page_content for doc in results])
        return {"pdf_content": content}
    return {"pdf_content": "Error: Document knowledge base not initialized."}

pdf_retrieval_tool = Tool(
    name="pdf_retrieval_tool",
    description="Useful for answering questions from the bank's internal knowledge base (PDFs). Input should be a specific question.",
    func=retrieve_from_pdf,
    args_schema=PDFRetrievalInput,
)

def search_web_func(query: str) -> dict:
    """Performs web search and returns structured tool output."""
    try:
        search_docs = tavily_search.invoke(query)
        formatted_docs = "\n\n---\n\n".join(
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs.get("results", [])
        )
        return {"web_content": formatted_docs or "No results found from web search."}
    except Exception as e:
        print(f"Web search error: {e}")
        return {"web_content": f"Error during web search: {e}"}

tavily_search_tool = Tool(
    name="tavily_search_tool",
    description="Useful for general questions or questions requiring up-to-date information from the web. Input should be a concise search query.",
    func=search_web_func,
    args_schema=WebSearchInput,
)

# Initialize SQL Agent (Primary Method)
SQL_AGENT = None
if db:
    try:
        SQL_TOOLKIT = SQLDatabaseToolkit(db=db, llm=llm)
        SQL_SYSTEM_PROMPT = """You are an agent designed to interact with a SQL database. Given an input question, create a syntactically correct {dialect} query, execute it, and return the answer.
        - You must query only the necessary columns.
        - You must double-check your query before execution.
        - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP).
        - ALWAYS look at the tables first to understand the schema."""
        
        SQL_AGENT = create_react_agent(
            llm,
            SQL_TOOLKIT.get_tools(),
            prompt=SQL_SYSTEM_PROMPT.format(dialect=db.dialect),
        )
        print("SQL Agent initialized successfully.")
    except Exception as e:
        print(f"Error initializing SQL Agent: {e}. SQL query tool will not be available.")

def execute_sql_query_func(query: str) -> dict:
    """Executes a SQL query using the pre-initialized SQL agent and returns the result."""
    if not SQL_AGENT:
        return {"sql_result": "Error: SQL Agent not initialized."}
    try:
        response_generator = SQL_AGENT.stream(
            {"messages": [HumanMessage(content=query)]}, stream_mode="values"
        )
        full_response_content = []
        for chunk in response_generator:
            if 'messages' in chunk and chunk['messages']:
                content = chunk['messages'][-1].content
                if content:
                    full_response_content.append(content)
        
        result = "\n".join(full_response_content) if full_response_content else "No response from SQL agent."
        return {"sql_result": result}
    except Exception as e:
        return {"sql_result": f"Error executing SQL query: {e}"}

sql_query_tool = Tool(
    name="sql_query_tool",
    description="Useful for answering questions requiring data from a SQL database (e.g., 'How many users are there?'). Input should be a natural language question.",
    func=execute_sql_query_func,
    args_schema=SQLQueryInput,
)

# Combine all tools (Corrected logic)
tools = [pdf_retrieval_tool, tavily_search_tool,sql_query_tool]
if SQL_AGENT:
    tools.append(sql_query_tool)

# ==========================
# 🧠 Graph Nodes
# ==========================

def get_time_based_greeting():
    """Return an appropriate greeting based on the current time."""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12: return "Good morning"
    if 12 <= current_hour < 17: return "Good afternoon"
    return "Good evening"

def agent_node(state: State):
    """
    The Router Node: Decides whether to call a tool or generate a final answer.
    This node's prompt is focused on routing, not on generating the final answer.
    """
    print("--- AGENT NODE (ROUTER) ---")
    messages = state["messages"]
    
    # Handle the very first message with a greeting
    if len(messages) == 1:
        return {"messages": [AIMessage(content=f"{get_time_based_greeting()}! I am Damilola, your AI-powered virtual assistant. Welcome to ATB Bank. How can I help you today?")]}
    
    system_prompt = SystemMessage(
        content=f"""You are Damilola, a helpful AI assistant for ATB Bank. Your role is to decide the next step in the conversation.
        
        You have access to the following tools: {', '.join([t.name for t in tools])}.
        
        1. Review the user's latest message in the context of the conversation history.
        2. If the user's question can be answered using one of your tools, call the most appropriate tool with the correct input.
        3. If you have already used a tool and have enough information to answer the user's question, respond directly.
        4. Do not generate the final answer here. Your job is to either call a tool or indicate that you're ready to answer.
        
        Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
    )
    
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([system_prompt] + messages)
    
    return {"messages": [response]}

def generate_final_answer_node(state: State):
    """
    The Generator Node: Creates the final structured answer after gathering all necessary context from tools.
    """
    print("--- GENERATE FINAL ANSWER NODE ---")
    user_query = state["messages"][-1].content
    
    context_parts = []
    if state.get("pdf_content"): context_parts.append(f"PDF Content:\n{state['pdf_content']}")
    if state.get("web_content"): context_parts.append(f"Web Content:\n{state['web_content']}")
    if state.get("sql_result"): context_parts.append(f"SQL Database Result:\n{state['sql_result']}")
    if state.get("attached_content"): context_parts.append(f"Attached Content:\n{state['attached_content']}")
    context = "\n\n".join(context_parts) if context_parts else "No additional context was retrieved."

    # Prompt designed to generate a structured JSON output based on the Answer schema
    prompt = f"""You are Damilola, the AI-powered virtual assistant for ATB Bank.
    Your goal is to provide a final, comprehensive, and empathetic answer based on the user's question and the context gathered from your tools.
    
    User Question: "{user_query}"
    
    Available Context:
    ---
    {context}
    ---
    
    Based on all the information above, generate a structured response. You MUST format your response as a JSON object that strictly follows the schema below.

    Schema:
    {{
      "answer": "str: A clear, concise, empathetic, and polite response directly addressing the user's question. Use straightforward language.",
      "sentiment": "int: An integer rating of the user's sentiment, from -2 (very negative) to +2 (very positive).",
      "ticket": "List[str]: A list of service channels relevant to any unresolved issue. Possible values: ['POS', 'ATM', 'Web', 'Mobile App', 'Branch', 'Call Center', 'Other']. Leave empty if not applicable.",
      "source": "List[str]: A list of sources used. Possible values: ['PDF Content', 'Web Search', 'SQL Database', 'User Provided Context', 'Internal Knowledge']. Leave empty if no specific source was used."
    }}
    """
    
    structured_llm = llm.with_structured_output(Answer)
    final_answer_obj = structured_llm.invoke(prompt)
    
    # Append the human-readable part of the answer to the message history
    new_messages = state["messages"] + [AIMessage(content=final_answer_obj.answer)]
    print ("Ajaiye",final_answer_obj)
    
    return {
        "final_answer": final_answer_obj,
        "messages": new_messages
    }

def summarize_conversation(state: State):
    """Generates a final summary of the conversation."""
    print("--- SUMMARIZE CONVERSATION NODE ---")
    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
    
    summarize_prompt = f"""Please provide a structured summary of the following conversation.
    
    Conversation History:
    {conversation_history}
    
    Provide the output in a JSON format matching this schema:
    {{
        "summary": "A concise summary of the entire conversation.",
        "sentiment": "Overall sentiment score of the conversation (-2 to +2).",
        "unresolved_tickets": ["List of channels with unresolved issues."],
        "all_sources": ["All unique sources referenced in the conversation."]
    }}
    """
    
    structured_llm = llm.with_structured_output(Summary)
    summary_obj = structured_llm.invoke(summarize_prompt)
    
    # Create the final metadata dictionary for logging
    final_answer = state.get("final_answer")
    metadata_dict = {
        "question": state["messages"][-2].content if len(state["messages"]) > 1 else "",
        "answer": final_answer.answer if final_answer else "N/A",
        "sentiment": final_answer.sentiment if final_answer else 0,
        "ticket": final_answer.ticket if final_answer else [],
        "source": final_answer.source if final_answer else [],
        "summary": summary_obj.summary,
        "summary_sentiment": summary_obj.sentiment,
        "summary_unresolved_tickets": summary_obj.unresolved_tickets,
        "summary_sources": summary_obj.all_sources,
    }

    return {
        "conversation_summary": summary_obj,
        "metadatas": metadata_dict
    }

# Define the condition to check if a tool was called
def should_continue(state: State) -> str:
    """Determines the next step: call a tool or generate the final answer."""
    if state["messages"][-1].tool_calls:
        return "tools"
    return "generate_final_answer"

# ==========================
# 🔄 Graph Workflow
# ==========================

def build_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(State)
    
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tools", ToolNode(tools=tools))
    workflow.add_node("generate_final_answer", generate_final_answer_node)
    workflow.add_node("summarize", summarize_conversation)





    workflow.set_entry_point("agent_node")
    workflow.add_conditional_edges("agent_node", should_continue)
    workflow.add_edge("tools", "agent_node")
    workflow.add_edge("generate_final_answer", "summarize")
    workflow.add_edge("summarize", END)
    
    # Initialize checkpointing with a robust fallback
    memory = None
    try:
        # DB_FILE_PATH should be defined, e.g., "checkpoints.sqlite"
        conn = sqlite3.connect(DB_FILE_PATH, check_same_thread=False)
        memory = SqliteSaver(conn=conn)
        print("SQLite checkpointing connected successfully.")
    except Exception as e:
        
        print(f"Error connecting to SQLite for checkpointing: {e}. Using in-memory saver.")
        memory = InMemorySaver()

    return workflow.compile(checkpointer=memory)

   


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
# Main processing function
def process_message(message_content: str, session_id: str, file_path: Optional[str] = None):
    """Main function to process user messages using the LangGraph agent."""
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    attached_content = None # Simplified for this example
    # Image processing logic can be added here as in the original code
       # Only process image if file_path is provided
    # if file_path and os.path.exists(file_path):
    if file_path:
        try:
            image = Image.open(file_path)
            image.thumbnail([512, 512]) # Resize for efficiency
            prompt = "Describe the content of the picture in detail."
            configure(api_key=GOOGLE_API_KEY)
            # genai.configure(api_key=GOOGLE_API_KEY)  # Configure the API key
            # modelT = genai.GenerativeModel('gemini-pro-vision') # Specify the vision model
            modelT = GenerativeModel(model_name="gemini-2.0-flash", generation_config={"temperature": 0.7,"max_output_tokens": 512 })
            response = modelT.generate_content([image, prompt])
            attached_content = response.text
            print ("Attached content from image:", attached_content)
        except Exception as e:
            print(f"Error processing image attachment: {e}")
            attached_content = f"Error: Could not process attached file ({e})"
    elif file_path:
        print(f"Warning: Attached file not found at {file_path}. Skipping image processing.")
    
    initial_state = {"messages": [HumanMessage(content=message_content)], "attached_content": attached_content}
    
    output = graph.invoke(initial_state, config)
    print("--- LangGraph workflow completed ---")
    
    # Extract final answer from the structured Pydantic object
    final_answer_obj = output.get('final_answer')
    final_answer_content = final_answer_obj.answer if final_answer_obj else "No final answer was generated."
    print("--- LangGraph Akule ---",final_answer_content)

    return {
        "answer": final_answer_content,
        "metadata": output.get("metadatas", {})
    }


