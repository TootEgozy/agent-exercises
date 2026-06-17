from pathlib import Path
from rich.console import Console
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import json

load_dotenv(override=True)
root = Path(__file__).resolve().parent
todos = []
completed = []
openai = OpenAI()
claude = Anthropic()

def show(txt):
    try:
        Console().print(txt)
    except Exception:
        print(txt)

def get_todo_report():
    result = "" 
    for index, todo in enumerate(todos):
        if completed[index]:
            result += f"Todo #{index + 1}: [green][strike]{todo}[/strike][/green]\n"
        else:
            result += f"todo #{index + 1}: {todo}\n"
    show(result)
    return result


def create_todos(descriptions):
    todos.extend(descriptions)
    completed.extend([False] * len(descriptions))
    return get_todo_report()


def mark_complete(index, completion_notes):
    if 1 <= index <= len(todos):
        completed[index - 1] = True
    else:
        return "No todo in this index."
    Console().print(completion_notes)
    return get_todo_report()


create_todos_json = {
    "name": "create_todos",
    "description": "Add new todos to a list from a list of descriptions and return the full todos list",
    "parameters": {
        "type": "object",
        "required": ["descriptions"],
        "additionalProperties": False,
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "title": "Descriptions"
            }
        }
    }
}


mark_complete_json = {
    "name": "mark_complete",
    "description": "Mark complete the todo at the given position (starting from 1) and return the string of todos state",
        "parameters": {
        'properties': {
            'index': {
                'description': 'The 1-based index of the todo to mark as complete',
                'title': 'Index',
                'type': 'integer'
                },
            'completion_notes': {
                'description': 'Notes about how you completed the todo in rich console markup',
                'title': 'Completion Notes',
                'type': 'string'
                }
            },
        'required': ['index', 'completion_notes'],
        'type': 'object',
        'additionalProperties': False
    }
}

tools = [{"type": "function", "function": create_todos_json},
        {"type": "function", "function": mark_complete_json}]

allowed_tools = {"create_todos": create_todos, "mark_complete": mark_complete}

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = allowed_tools.get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    return results

def loop(messages):
    global tools
    done = False
    while not done:
        response = openai.chat.completions.create(model='gpt-5.2', messages=messages, tools=tools, reasoning_effort="none")
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True
    show(response.choices[0].message.content)

def handle_questions_cache(summary):
    global root
    with open(root / "data" / "todo.json", "r") as file:
        cache = json.load(file)
    if cache["count"] > 10:
        cache["prev_questions"] = []
        cache["count"] = 0
    else:
        cache["prev_questions"].append(summary)
        cache["count"] += 1
    with open(root / "data" / "todo.json", "w") as file:
        json.dump(cache, file, indent=4)



def get_question():
    global root
    prev_questions = []
    with open(root / "data" / "todo.json", "r") as file:
        cache = json.load(file)
        prev_questions = cache["prev_questions"]
        summary_message = [{"role": "user", "content": f"Provide a riddle, a logical or a mathematical question to solve. Reply with just the question in simple text, no additions. avoid repeating these riddles (summarised): {prev_questions}"}]
    response = claude.messages.create(model="claude-sonnet-4-5", messages = summary_message, max_tokens=1000)
    question = response.content[0].text
    summary = openai.chat.completions.create(model='gpt-5.2', messages=[{"role": "user", "content": f"Your task is to summarize and shorten a question or a riddle for caching purposes. reply only with the summary in plain text. the question: {question}"}], 
    reasoning_effort="none")
    handle_questions_cache(summary.choices[0].message.content)
    show(f"[blue]{question}[/blue]\n")
    return question


openai_system_message = """
You are given a problem to solve, by using your todo tools to plan a list of steps, then carrying out each step in turn.
Now use the todo list tools, create a plan, carry out the steps, and reply with the solution.
If any quantity isn't provided in the question, then include a step to come up with a reasonable estimate.
Provide your solution in Rich console markup without code blocks.
Do not ask the user questions or clarification; respond only with the answer after using your tools.
"""
question = get_question()
messages = [{"role": "system", "content": openai_system_message}, {"role": "user", "content": question}]

loop(messages)