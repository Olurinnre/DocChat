import os
from pathlib import Path

import groq
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
c = groq.Groq(api_key=os.getenv("LLM_API_KEY"))
try:
    r = c.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": "Say hi"}],
    )
    print(r.choices[0].message.content)
except groq.APIStatusError as e:
    print(e.status_code, e.response.text)