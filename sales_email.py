from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, function_tool, OpenAIChatCompletionsModel, output_guardrail, GuardrailFunctionOutput
import os
from pydantic import BaseModel, Field
import resend
from email.message import EmailMessage
import asyncio

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
mail_address = os.getenv("RESEND_MAIL_ADDRESS")
resend.api_key = os.environ["RESEND_API_KEY"]

gemini_client = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=google_api_key)
openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
groq_client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)

gemini_model = OpenAIChatCompletionsModel(model="gemini-3.1-flash-lite", openai_client=gemini_client)
kimi_model = OpenAIChatCompletionsModel(model="moonshotai/kimi-k2.6", openai_client=openrouter_client)
oss_model = OpenAIChatCompletionsModel(model="openai/gpt-oss-120b", openai_client=groq_client)

instructions = """
You are a sales agent working for BuildMind,
a company that provides an AI-powered SaaS tool for diagnosing CI/CD pipeline failures, identifying root causes, and suggesting fixes — so engineering teams spend less time debugging broken builds and more time shipping.
You write short, compelling sales emails that are likely to get a response.
"""

sales_agent1 = Agent(name="Gemini Sales Agent", instructions=instructions, model=gemini_model)
sales_agent2 = Agent(name="Kimi Sales Agent", instructions=instructions, model=kimi_model)
sales_agent3 = Agent(name="GPT-OSS Sales Agent",instructions=instructions, model=oss_model)

description = "Use this tool to write a sales email. In the input, just instruct it to write a sales email."

tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

class EmailReview(BaseModel):
    is_professional: bool = Field(description="Whether the email is professional and appropriate")
    number_of_sentences: int = Field(description="The number of sentences in the body of the email, not including the greeting and signature")
    contains_placeholders: bool = Field(description="Whether the email contains placeholders for personalization")
    spam_score: int = Field(description="Whether the language and phrases in the email might be labeled as spam, score from 1 to 100")

checker = Agent(name="Checker", instructions="You review potential sales emails", model=gemini_model, output_type=EmailReview)
review_email_tool = checker.as_tool(tool_name="review_email_tool", tool_description="Use this tool to evaluate sales email")

def send_email(subject, text_body, html_body):
    resend.Emails.send({
        "from": "onboarding@resend.dev",    
        "to": mail_address,  # type: ignore
        "subject": subject,
        "html": html_body,
        "text": text_body
    })


@function_tool
def send_email_tool(subject: str, text_body: str, html_body: str) -> str:
    """
    Send out an email with the given subject and body to all sales prospects
    
    Args:
        subject: The subject of the email
        text_body: The body of the email as plain text
        html_body: The HTML body of the email
    """
    send_email(subject, text_body, html_body)
    return "Email sent successfully"

tools = [tool1, tool2, tool3, review_email_tool, send_email_tool]

instructions = "You are a Sales Manager at BuildMind. Your goal is to find the single best cold sales email using the sales_writer tools."
task = """
Follow these steps:

1. Generate Drafts: Use each of the three sales_email_writer tools to generate different email drafts.
Just instruct each to write a sales email; no further details are needed.
Do not proceed until all three drafts are ready, one from each tool.
 
2. Evaluate and Select: Review the drafts using your tool and choose the single best email using your judgment of which one is most effective, and the tool's output parameters.
 
3. Use your tool to send the best email (and only the best email) to the user. Only send 1 email.
"""

sales_manager = Agent(name="Sales Manager", instructions=instructions, tools=tools, model="gpt-5.4-mini")

async def main():
    with trace("Sales Manager across different models"):
        result = await Runner.run(sales_manager, task)
    print(result.final_output)

asyncio.run(main())


