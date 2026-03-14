# Run this once to verify everything works
# save as: test_setup.py

from core.config import OPENAI_API_KEY, LLM_MODEL
from langchain_openai import ChatOpenAI

def test_connection():
    print(f"Model: {LLM_MODEL}")
    print(f"API Key loaded: {'Yes' if OPENAI_API_KEY else 'NO - CHECK YOUR .env FILE'}")
    
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
    response = llm.invoke("Say: setup successful")
    print(f"LLM Response: {response.content}")

if __name__ == "__main__":
    test_connection()