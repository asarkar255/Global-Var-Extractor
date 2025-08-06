import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# -----------------------------
# Load Knowledge Base
# -----------------------------
ruleset_loader = TextLoader("ruleset.txt")
documents = ruleset_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

abap_exmpl_loader = TextLoader("abap_program.txt")
exmpl_abap = abap_exmpl_loader.load()
text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs2 = text_splitter2.split_documents(exmpl_abap)

all_docs = docs + docs2

# -----------------------------
# Embeddings + Vector Store
# -----------------------------
persist_directory = "./chroma_db"
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.2}
)

# -----------------------------
# LLM and Prompt
# -----------------------------
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

remediate_prompt = PromptTemplate(
    input_variables=["Rules",  "example_rules", "input_code"],
    template="""
You are an SAP ABAP Remediation Expert.
Your task is to fully remediate all forms and subroutines in the ECC ABAP code.
DO NOT skip any section or write placeholders like "...rest is similar".
Comment out old code and insert new code following clean S/4HANA standards.

Apply the following:
- Comment legacy TABLES, OCCURS, LIKE, etc.
- Replace with DATA, TYPES, and modern SELECT.
- Follow all remediation rules strictly.
- Follow syntax and formatting exactly like examples.
- Ensure final output is complete and not trimmed.

Rules:
{Rules}

Example Rules:
{example_rules}

ECC ABAP Code:
{input_code}

Output:
[Remediated ABAP Code]
"""
)

# -----------------------------
# Memory Management
# -----------------------------
chat_histories = {}
def memory_factory(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

remediate_chain = RunnableWithMessageHistory(
    remediate_prompt | llm | StrOutputParser(),
    memory_factory,
    input_messages_key="input_code",
    history_messages_key="history"
)

# -----------------------------
# Input Schema
# -----------------------------
class ABAPCodeInput(BaseModel):
    code: str

# -----------------------------
# Global Declarations Extractor
# -----------------------------
def extract_global_declarations(remediated_code: str) -> str:
    lines = remediated_code.splitlines()
    global_blocks = []
    capture = False
    block = ""

    start_keywords = [
        "DATA", "DATA:",
        "TYPES", "TYPES:",
        "CONSTANTS", "CONSTANTS:",
        "TABLES", "TABLES:",
        "PARAMETERS", "PARAMETERS:",
        "SELECT-OPTIONS", "SELECT-OPTIONS:"
    ]

    for line in lines:
        stripped_line = line.strip()

        # Skip empty or fully commented lines
        # if not stripped_line or stripped_line.startswith("*") or stripped_line.startswith('"'):
        if not stripped_line  or stripped_line.startswith('"'):
            continue

        # Normalize for matching
        upper_line = stripped_line.upper()

        # Strip inline comment for declaration end check
        code_only = re.split(r'"|\*', upper_line)[0].strip()

        if any(code_only.startswith(keyword) for keyword in start_keywords):
            capture = True
            block = line
            if code_only.endswith("."):
                global_blocks.append(block.strip())
                block = ""
                capture = False
        elif capture:
            block += "\n" + line
            if code_only.endswith("."):
                global_blocks.append(block.strip())
                block = ""
                capture = False

    if not global_blocks:
        return "[GLOBAL_DATA_START]\n[GLOBAL_DATA_END]"

    return "[GLOBAL_DATA_START]\n" + "\n\n".join(global_blocks) + "\n[GLOBAL_DATA_END]"

# -----------------------------
# Main Remediation + Extraction Logic
# -----------------------------
def extract_globals_from_remediated_code(input_code: str):
    rules_text = "\n\n".join([doc.page_content for doc in docs])
    example_rules_text = "\n\n".join([doc.page_content for doc in docs2])

    lines = input_code.splitlines()
    chunks = [lines[i:i + 800] for i in range(0, len(lines), 800)]

    full_remediated_code = ""

    for chunk_lines in chunks:
        chunk_code = "\n".join(chunk_lines)

        response = remediate_chain.invoke(
            {
                "Rules": rules_text,
                "example_rules": example_rules_text,
                "input_code": chunk_code
            },
            config={"configurable": {"session_id": "extract_globals"}}
        )

        full_remediated_code += response

    global_context = extract_global_declarations(full_remediated_code)

    return global_context

# -----------------------------
# FastAPI Endpoint
# -----------------------------
@app.post("/remediate_and_extract_globals/")
async def remediate_and_extract_globals(input_data: ABAPCodeInput):
    result = extract_globals_from_remediated_code(input_data.code)
    return {"global_data": result}
