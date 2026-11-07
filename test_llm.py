from dotenv import load_dotenv
load_dotenv()
import os
import openai

c = openai.OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
try:
    r = c.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": "Say hi"}],
    )
    print(r.choices[0].message.content)
except openai.APIStatusError as e:
    print(e.status_code, e.response.text)