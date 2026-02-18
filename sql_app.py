import streamlit as st
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_groq import ChatGroq
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


st.set_page_config(page_title="SQL ChatBot", page_icon="🤖")
st.title("LangChain: Chat with SQL DB")

# ---------------- DB SETTINGS ----------------

radio_opt = ["Use Local SQLite DB", "Use MySQL DB"]
select_opt = st.sidebar.radio("Choose the DB", radio_opt)

if select_opt == "Use MySQL DB":
    host = st.sidebar.text_input("Host", value="localhost")
    user = st.sidebar.text_input("User", value="root")
    password = st.sidebar.text_input("Password", type="password")
    database = st.sidebar.text_input("Database", value="mydb")

    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{user}:{password}@{host}/{database}"
    )
    st.success("✅ MySQL database connected!")

else:
    db = SQLDatabase.from_uri("sqlite:///local.db")
    st.success("✅ Local SQLite database connected!")

# Show schema (helps agent)
with st.expander("📂 Database schema"):
    st.write(db.get_table_info())

# ---------------- MODEL SETTINGS ----------------

with st.sidebar:
    st.header("Model Settings")

    LANGCHAIN_API_KEY = st.text_input("LANGCHAIN_API_KEY", type="password")
    GROQ_API_KEY = st.text_input("GROQ_API_KEY", type="password")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_token = st.slider("Max Tokens", 1, 2048, 512)
    model_name = st.selectbox(
        "Model Name",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    )

# ---------------- ENV ----------------

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "SQL ChatBot"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

# ---------------- LLM ----------------

model = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_token,
    api_key=GROQ_API_KEY,
    streaming=True,
)

st.success("🤖 Model loaded successfully!")

# ---------------- SQL AGENT ----------------

toolkit = SQLDatabaseToolkit(db=db, llm=model)

agent = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True,
    agent_type="tool-calling",
    max_iterations=40,
    early_stopping_method="generate",
    max_execution_time=120,
    handle_tool_error=True,
)


# ---------------- CHAT MEMORY ----------------

if "messages" not in st.session_state or st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [{
        "role": "system",
        "content": "You are a SQL assistant. Generate correct SQL in minimal steps. Avoid retries."
    }]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input(placeholder="Ask a question about the database")

# ---------------- CHAT LOOP ----------------

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        streamlit_callback= StreamlitCallbackHandler(st.container())
        try:
#            response = agent.run( {"input": prompt, "streamlit_callback": streamlit_callback} )
            response = agent.run(prompt, callbacks=[streamlit_callback])

            #output = response.get("output", str(response))
            st.write(response)

        except Exception as e:
            response = f"⚠️ Error: {str(e)}"
            st.write(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
            )
