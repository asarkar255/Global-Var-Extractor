# Filename: main.py (FastAPI app with variable mapping)

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
# Global Mapping Extractor
# -----------------------------
def extract_variable_mapping(remediated_code: str) -> list:
    mapping = []
    lines = remediated_code.splitlines()

    original_vars = {}
    new_vars = {}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("*TABLES:"):
            tables = re.findall(r'TABLES:\s*(.*)\.', stripped)
            if tables:
                for tbl in tables[0].split(','):
                    tbl_name = tbl.strip()
                    if tbl_name:
                        original_vars[tbl_name] = f"gs_{tbl_name}"
        elif "OCCURS 0 WITH HEADER LINE" in stripped:
            match = re.match(r"\*?\s*DATA:\s+BEGIN OF (\w+).*", stripped)
            if match:
                name = match.group(1)
                original_vars[name] = f"gs_{name}"

    for line in lines:
        m = re.match(r"\s*DATA:\s+(\w+)\s+TYPE\s+(\w+).*", line)
        if m:
            var, typ = m.groups()
            if var.startswith("gs_"):
                new_vars[typ] = var

    for old, new in original_vars.items():
        if old in new_vars:
            mapping.append({
                "original": old,
                "replacement": new_vars[old],
                "note": f"Replace all usage of {old} as work area with {new_vars[old]}"
            })
        else:
            mapping.append({
                "original": old,
                "replacement": new,
                "note": f"Replace all usage of {old} as work area with {new}"
            })

    return mapping

# -----------------------------
# Injection Helper
# -----------------------------
def inject_variable_mapping_block(code: str, mapping: list) -> str:
    start_tag = "[GLOBAL_DATA_START]"
    end_tag = "[GLOBAL_DATA_END]"

    lines = code.splitlines()
    try:
        start_idx = lines.index(start_tag)
        end_idx = lines.index(end_tag)
    except ValueError:
        return code

    mapping_lines = ["[VARIABLE_MAPPING]"]
    for item in mapping:
        mapping_lines.append(
            f"Original: {item['original']}   -> Replacement: {item['replacement']}    -- {item['note']}"
        )
    mapping_lines.append("[VARIABLE_MAPPING_END]")

    # Inject between start and end
    new_global_block = lines[start_idx+1:end_idx] + [""] + mapping_lines
    new_code = lines[:start_idx+1] + new_global_block + lines[end_idx:]

    return "\n".join(new_code)

# -----------------------------
# Main Logic
# -----------------------------
def extract_globals_with_mapping(input_code: str):
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

    mapping = extract_variable_mapping(full_remediated_code)

    # Inject mapping inside global block
    final_code = inject_variable_mapping_block(full_remediated_code, mapping)

    return {
        "remediated_code": final_code,
        "variable_mapping": mapping
    }

# -----------------------------
# FastAPI Endpoint
# -----------------------------
@app.post("/remediate_and_extract_globals/")
async def remediate_and_extract_globals(input_data: ABAPCodeInput):
    result = extract_globals_with_mapping(input_data.code)
    return result
