import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
load_dotenv()

# This should print 'DEBUG: Key found: gsk_q...'
# Initialize the ChatGroq model
# Pro Tip: Set your key in your terminal/env so it's not in the code!
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0, 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Define the prompt
prompt = ChatPromptTemplate.from_template("Summarize this article in one sentence: {article}")

# Output parser
parser = StrOutputParser()

# Custom formatting function
def simple_formatter(text):
    return f"SUMMARY: {text}"

# The Chain (Optimized)
chain = prompt | llm | parser | RunnableLambda(simple_formatter)

# The Data
longArticle = """There was nothing so _very_ remarkable in that; nor did Alice think it
so _very_ much out of the way to hear the Rabbit say to itself, “Oh
dear! Oh dear! I shall be late!”... [rest of your text]"""

# 6. Execution
try:
    # This matches the {article} variable in your prompt template
    summary = chain.invoke({"article": longArticle})
    print(summary)
except Exception as e:
    print(f"Error: {e}")