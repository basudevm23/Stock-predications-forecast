# llm_gemini.py — correct for Gemini 2.5 Flash
import os
import google.generativeai as genai

def ask_gemini_for_actions(prompt: str) -> str:
    """
    Calls Gemini API to generate stock buy/sell recommendations
    based on recent ML predictions and stock data.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found. Set it in your environment first.")

    # Configure Gemini
    genai.configure(api_key=api_key)

    # Use Gemini 2.5 Flash model
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Generate response
    response = model.generate_content(prompt)

    # Safely extract text
    try:
        return response.text.strip()
    except Exception:
        return "⚠️ Gemini did not return a valid text response."
