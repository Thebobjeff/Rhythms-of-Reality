import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredMarkdownLoader

load_dotenv()

# 1. Load the Markdown file
# Make sure 'article.md' is in the same folder as your main.py!
loader = UnstructuredMarkdownLoader("./data/alice-in-wonder-land.md")
docs = loader.load()

# 2. Extract the text content from the Document object
# loader.load() returns a list of Documents; we grab the first one's content
markdown_content = docs[0].page_content

# 3. Setup the rest of your chain as before
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
prompt = ChatPromptTemplate.from_template("Summarize this Markdown file in one sentence: {article}")
parser = StrOutputParser()

chain = prompt | llm | parser

# 4. Invoke using the file content
try:
    summary = chain.invoke({"article": markdown_content})
    print(f"SUCCESS: {summary}")
except Exception as e:
    print(f"ERROR: {e}")