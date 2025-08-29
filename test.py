import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def test_llm():
    """
    A standalone script to test the connection to the Google Gemini LLM.
    This bypasses all other parts of the application (FastAPI, CrewAI, etc.)
    for a direct and simple connection test.
    """
    print("--- 🚀 Starting Standalone LLM Connection Test ---")

    # 1. Load environment variables from .env file
    print("--- 📄 Loading .env file...")
    load_dotenv()
    print("--- ✅ .env file loaded.")

    # 2. Get the API Key
    print("--- 🔑 Retrieving GOOGLE_API_KEY...")
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\n--- ❌ CRITICAL ERROR: GOOGLE_API_KEY not found! ---")
        print("--- Please ensure you have a .env file in the same directory ---")
        print("--- with the line: GOOGLE_API_KEY=your_actual_api_key ---\n")
        return

    # Use a placeholder to avoid logging the full key
    print(f"--- ✅ API Key found (starts with: '{api_key[:4]}...').")

    # 3. Initialize the LLM
    try:
        print("--- 🧠 Initializing Google Generative AI client...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key
        )
        print("--- ✅ LLM client initialized successfully.")
    except Exception as e:
        print(f"\n--- ❌ CRITICAL ERROR: Failed to initialize the LLM client. ---")
        print(f"--- Error Details: {e} ---\n")
        return

    # 4. Send a test query
    try:
        print("--- 💬 Sending a test prompt to the LLM: 'Hello! Who are you?'...")
        response = llm.invoke("Hello! In one short sentence, who are you?")

        print("\n" + "=" * 50)
        print("🎉 SUCCESS! The LLM is working correctly. 🎉")
        print(f"🤖 LLM Response: {response.content}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\n" + "=" * 50)
        print(f"--- ❌ CRITICAL ERROR: The LLM call failed. ---")
        print(f"--- This might be due to an invalid API key or network issue. ---")
        print(f"--- Error Details: {e} ---")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    test_llm()