import os
import google.generativeai as genai

# Check if GEMINI_API_KEY is set
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Gemini API key is NOT set! Please run:")
    print("   setx GEMINI_API_KEY \"your_api_key_here\"   (Windows)")
    print("Then restart your terminal and try again.")
else:
    print(f"✅ Found Gemini API key (first 8 chars): {api_key[:8]}********")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("Say 'Gemini API key test successful!'")
        print("🟢 Gemini connection test passed!")
        print("Model Response:", response.text)
    except Exception as e:
        print("❌ Error connecting to Gemini API:")
        print(e)
