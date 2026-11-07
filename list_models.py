from dotenv import load_dotenv
load_dotenv()
import os
import openai

c = openai.OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
for m in sorted(c.models.list().data, key=lambda m: m.id):
    print(m.id)