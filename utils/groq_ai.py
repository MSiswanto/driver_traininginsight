import os
from groq import Groq

def generate_groq_insight(text: str) -> str:
    """
    Generate racing insight using LLaMA 3.3 on Groq Cloud.
    """

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", #llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert race telemetry analyst."},
                {"role": "user", "content": text}
            ],
            max_tokens=600,
            temperature=0.4
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ AI Insight Error: {e}"
