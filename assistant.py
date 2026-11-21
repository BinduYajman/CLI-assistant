from pydantic import BaseModel, ConfigDict, ValidationError
from typing import Literal
import argparse
import os
import json
import re
import requests
from functools import lru_cache
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import (
    Groq,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    APIConnectionError,
    APIStatusError
)

total_cost = 0.0

@lru_cache(maxsize=None)
def get_model_pricing(model_id):
    url = "https://console.groq.com/docs/models"

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    page_text = soup.get_text(separator=" ", strip=True)

    model_index = page_text.find(model_id)

    if model_index == -1:
        raise ValueError(f"Model {model_id} not found.")

    model_text = page_text[model_index:model_index + 300]

    match = re.search(
        r"\$([\d.]+)\s*input\s*\$([\d.]+)\s*output",
        model_text
    )

    if not match:
        raise ValueError(f"Pricing information unavailable for {model_id}.")

    input_price = float(match.group(1))
    output_price = float(match.group(2))

    return input_price, output_price

def calculate_cost(input_tokens, output_tokens, input_price, output_price):
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    total_cost = input_cost + output_cost

    return total_cost

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Please add it to your .env file."
    )

client = Groq(api_key=api_key)

def handle_api_error(error):

    if isinstance(error, AuthenticationError):
        print("Error: Invalid Groq API key.")

    elif isinstance(error, RateLimitError):
        print("Error: API rate limit exceeded. Try again later.")

    elif isinstance(error, NotFoundError):
        print("Error: Requested model or resource not found.")

    elif isinstance(error, APIConnectionError):
        print("Error: Unable to connect to Groq. Check your internet.")

    elif isinstance(error, APIStatusError):
        print(f"Error: Groq API returned status {error.status_code}.")

    else:
        print(f"Unexpected API error: {error}")

class TextSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main_topic: str
    key_entities: list[str]
    sentiment: Literal["positive", "neutral", "negative"]
    one_line_summary: str

def ask_groq(question, model):

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],
            stream=True
        )

        collected_content = []
        usage = None

        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content

                if content:
                    print(content, end="", flush=True)
                    collected_content.append(content)

            if chunk.usage:
                usage = chunk.usage

        print()
        assistant_reply = "".join(collected_content)

        global total_cost

        if usage:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

            input_price, output_price = get_model_pricing(model)

            cost = calculate_cost(
                input_tokens,
                output_tokens,
                input_price,
                output_price
            )

            total_cost += cost

            print("\n--- Token Usage ---")
            print(f"Input tokens: {input_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(f"Total tokens: {usage.total_tokens}")
            print(f"Estimated cost: ${cost:.8f}")
            print(f"Running total cost: ${total_cost:.8f}")

        return assistant_reply

    except (AuthenticationError, RateLimitError,
            NotFoundError, APIConnectionError, APIStatusError) as error:

        handle_api_error(error)
        return None

def structured_extract(text):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract information from the given text. "
                        "Identify its main topic, key entities, "
                        "sentiment, and a one-line summary."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "text_summary",
                    "strict": True,
                    "schema": TextSummary.model_json_schema()
                }
            }
        )

        result = json.loads(response.choices[0].message.content)

        return TextSummary.model_validate(result)
    
    except (AuthenticationError, RateLimitError,
            NotFoundError, APIConnectionError, APIStatusError) as error:

        handle_api_error(error)
        return None

    except json.JSONDecodeError:
        print("Error: The model returned invalid JSON.")
        return None

    except ValidationError as error:
        print("Error: The extracted data does not match the required schema.")
        print(error)
        return None
        

def load_history(session):
    filename = f"history_{session}.json"

    try:
        with open(filename, "r") as file:
            return json.load(file)

    except FileNotFoundError:
        return []

def save_history(messages, session):
    filename = f"history_{session}.json"

    with open(filename, "w") as file:
        json.dump(messages, file, indent=4)

def clear_history():

    if os.path.exists("history.json"):
        os.remove("history.json")
        print("Conversation history cleared successfully!")

    else:
        print("No conversation history found.")

def trim_history(messages, max_turns=20):

    system_messages = [
        message for message in messages
        if message["role"] == "system"
    ]

    conversation = [
        message for message in messages
        if message["role"] != "system"
    ]

    return system_messages + conversation[-(max_turns * 2):]

def interactive_chat(model,session):
    global total_cost

    messages = load_history(session)

    print("\nAI Assistant started!")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:

        user_input = input("You: ")

        if not user_input.strip():
            continue

        if user_input.lower() in ["quit", "exit"]:
            save_history(messages,session)
            print("Goodbye!")
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            response = client.chat.completions.create(
                model=model,
                messages=trim_history(messages),
                stream=True
            )

        except (AuthenticationError, RateLimitError,
                NotFoundError, APIConnectionError, APIStatusError) as error:

            handle_api_error(error)
            continue

        collected_content = []
        usage = None

        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content

                if content:
                    print(content, end="", flush=True)
                    collected_content.append(content)

            if chunk.usage:
                usage = chunk.usage

        print()

        assistant_reply = "".join(collected_content)

        if usage:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

            input_price, output_price = get_model_pricing(model)

            cost = calculate_cost(
                input_tokens,
                output_tokens,
                input_price,
                output_price
            )

            total_cost += cost

            print("\n--- Token Usage ---")
            print(f"Input tokens: {input_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(f"Total tokens: {usage.total_tokens}")
            print(f"Estimated cost: ${cost:.8f}")
            print(f"Running total cost: ${total_cost:.8f}")


        print("\nAssistant:", assistant_reply, "\n")

        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })

        save_history(messages,session)

def main():
    parser = argparse.ArgumentParser(
        description="CLI AI Assistant"
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="Ask a one-shot question"
    )

    parser.add_argument(
        "--system",
        type=str,
        help="Custom system prompt"
    )

    parser.add_argument(
    "--structured",
    type=str,
    metavar="TEXT",
    help="Extract structured information from text"
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear conversation history"
    )

    parser.add_argument(
    "--model",
    default="openai/gpt-oss-20b",
    help="Model to use for generating responses"
    )

    parser.add_argument(
    "--session",
    default="default",
    help="Name of the conversation session"
    )

    args = parser.parse_args()

    if args.clear:
        clear_history()

    elif args.structured:
        result = structured_extract(args.structured)
        if result is not None:
            print("\nStructured Output:")
            print(result.model_dump_json(indent=2))

    elif args.question:
        answer = ask_groq(args.question, args.model)
        if answer is not None:
            print("\nAssistant:", answer)

    else:
        interactive_chat(args.model, args.session)

    if args.system:
        print("Custom system prompt:", args.system)


if __name__ == "__main__":
    main()