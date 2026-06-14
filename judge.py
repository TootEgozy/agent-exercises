import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from IPython.display import Markdown, display
import ollama

load_dotenv(override=True)
ollama.pull('llama3.2:1b')

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

openai = OpenAI()
openai.model_name = "gpt-4.1-mini"
openai.name = "OpenAI"
claude = Anthropic()
claude.model_name = "claude-sonnet-4-5"
claude.name = "Claude"
gemini = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
gemini.model_name = "gemini-2.5-flash"
gemini.name = "Gemini"
deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
deepseek.model_name = "deepseek-chat"
deepseek.name = "DeepSeek"
groq = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
groq.model_name = "llama-3.3-70b-versatile"
groq.name = "Groq"
ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
ollama.model_name = "llama3.2:1b"
ollama.name = "Ollama"


competitors = [openai, claude, gemini, deepseek, groq, ollama]
answers = []


def ask_llm(llm, msg):
    messages = [{"role": "user", "content": msg}]
    if llm.model_name == "claude-sonnet-4-5":
        response = llm.messages.create(model=llm.model_name, messages=messages, max_tokens=1000)
        return response.content[0].text
    else:
        response = llm.chat.completions.create(model=llm.model_name, messages=messages)
        return response.choices[0].message.content

questioning = random.choice(competitors)
difficult_question = ask_llm(questioning, "Please come up with a challenging, nuanced question that I can ask a number of LLMs to evaluate their intelligence. Answer only with the question, no explanation. keep it short.")
print("_____________________________________")
print(f"{questioning.name} asking: {difficult_question}")
print("_____________________________________")

answer_prompt = "answer the following question, keep it short, no markup: /n" + difficult_question
for llm in competitors:
    answer = ask_llm(llm, answer_prompt)
    answers.append(answer)
    print(f"{llm.name}: {answer}")
    print("_____________________________________")


together = ""
for index, answer in enumerate(answers):
    together += f"# Response from competitor {index+1}\n\n"
    together += answer + "\n\n"

judge_prompt = f"""You are judging a competition between {len(competitors)} competitors.
Each was given this question:

{difficult_question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}
competitor numbers are integers only, no additional strings.

Here are the responses from each competitor:

{together}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""

judge = random.choice(competitors)
ruling_json = ask_llm(judge, judge_prompt)
results_dict = json.loads(ruling_json)
ranks = results_dict["results"]
print(f"{judge.name} rating:")
for index, result in enumerate(ranks):
    competitor = competitors[int(result)-1].name
    print(f"Rank {index+1}: {competitor}")