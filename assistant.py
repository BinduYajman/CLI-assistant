from pydantic import BaseModel, ConfigDict, ValidationError
from typing import Literal
import argparse
import os
import json
from dotenv import load_dotenv
from groq import (
    Groq,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    APIConnectionError,
    APIStatusError
)

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

def ask_groq(question):

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        usage = response.usage

        if usage:
            print("\n--- Token Usage ---")
            print(f"Input tokens: {usage.prompt_tokens}")
            print(f"Output tokens: {usage.completion_tokens}")
            print(f"Total tokens: {usage.total_tokens}")

        return response.choices[0].message.content

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
        

def load_history():
    try:
        with open("history.json", "r") as file:
            return json.load(file)

    except FileNotFoundError:
        return []

def save_history(messages):
    with open("history.json", "w") as file:
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

def interactive_chat():

    messages = load_history()

    print("\nAI Assistant started!")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:

        user_input = input("You: ")

        if not user_input.strip():
            continue

        if user_input.lower() in ["quit", "exit"]:
            save_history(messages)
            print("Goodbye!")
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=trim_history(messages)
            )

        except (AuthenticationError, RateLimitError,
                NotFoundError, APIConnectionError, APIStatusError) as error:

            handle_api_error(error)
            continue

        assistant_reply = response.choices[0].message.content

        print("\nAssistant:", assistant_reply, "\n")

        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })

        save_history(messages)

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

    args = parser.parse_args()

    if args.clear:
        clear_history()
    elif args.structured:
        result = structured_extract(args.structured)
        if result is not None:
            print("\nStructured Output:")
            print(result.model_dump_json(indent=2))

    elif args.question:
        answer = ask_groq(args.question)
        if answer is not None:
            print("\nAssistant:", answer)

    else:
        interactive_chat()

    if args.system:
        print("Custom system prompt:", args.system)


if __name__ == "__main__":
    main()