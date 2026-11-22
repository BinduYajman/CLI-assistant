import time
import argparse
import os
import json
import re
import requests
from functools import lru_cache
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError
from typing import Literal
from groq import (
    Groq,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    APIConnectionError,
    APIStatusError
)

total_cost = 0.0

MODEL_PRICING_FALLBACK = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "mixtral-8x7b-32768": (0.24, 0.24),
}

@lru_cache(maxsize=None)
def get_model_pricing(model_id):
    url = "https://console.groq.com/docs/models"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)

        model_index = page_text.find(model_id)

        if model_index != -1:
            model_text = page_text[model_index:model_index + 300]
            match = re.search(
                r"\$([\d.]+)\s*input\s*\$([\d.]+)\s*output",
                model_text
            )
            if match:
                input_price = float(match.group(1))
                output_price = float(match.group(2))
                return input_price, output_price
    except Exception:
        pass

    if model_id in MODEL_PRICING_FALLBACK:
        return MODEL_PRICING_FALLBACK[model_id]

    raise ValueError(f"Pricing information unavailable for {model_id}.")

def calculate_cost(input_tokens, output_tokens, input_price, output_price):
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    total_cost = input_cost + output_cost

    return total_cost

def track_cost(usage, model):
    global total_cost

    if not usage:
        print(
            f"Cost: unavailable | "
            f"Total session: ${total_cost:.8f}"
        )
        return

    try:
        input_price, output_price = get_model_pricing(model)

        cost = calculate_cost(
            usage.prompt_tokens,
            usage.completion_tokens,
            input_price,
            output_price
        )

        total_cost += cost

        print(
            f"Cost: ${cost:.8f} | "
            f"Total session: ${total_cost:.8f}"
        )

    except (requests.RequestException, ValueError) as error:
        print(f"Cost calculation unavailable: {error}")

_client = None

def get_client():
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Please add it to your .env file."
            )

        _client = Groq(api_key=api_key)
    return _client

def call_with_retry(model, messages, **kwargs):
    client = get_client()
    max_retries = 3

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )

        except (RateLimitError, APIConnectionError) as error:
            if attempt == max_retries - 1:
                raise error

            delay = 2 ** attempt
            print(f"\nRequest failed. Retrying in {delay} seconds...")
            time.sleep(delay)

        except APIStatusError as error:
            if error.status_code < 500 or attempt == max_retries - 1:
                raise error

            delay = 2 ** attempt
            print(f"\nServer error. Retrying in {delay} seconds...")
            time.sleep(delay)

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


def ask_groq(question, model, temperature=0.7, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    try:
        response = call_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True}
        )

        print("\nAssistant: ", end="", flush=True)

        collected_content = []
        usage = None

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                collected_content.append(content)

            if chunk.usage:
                usage = chunk.usage

        print("\n")
        assistant_reply = "".join(collected_content)

        track_cost(usage, model)

        return assistant_reply

    except (AuthenticationError, RateLimitError,
            NotFoundError, APIConnectionError, APIStatusError) as error:

        handle_api_error(error)
        return None


def structured_extract(text, model="openai/gpt-oss-20b"):
    try:
        response = call_with_retry(
            model=model,

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
        parsed = TextSummary.model_validate(result)

        if response.usage:
            track_cost(response.usage, model)

        return parsed
    
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

def clear_history(session="default"):
    filename = f"history_{session}.json"
    cleared = False

    if os.path.exists(filename):
        os.remove(filename)
        cleared = True

    # Also check legacy history.json if clearing default session
    if session == "default" and os.path.exists("history.json"):
        os.remove("history.json")
        cleared = True

    if cleared:
        print(f"Conversation history for session '{session}' cleared successfully!")
    else:
        print(f"No conversation history found for session '{session}'.")

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

def interactive_chat(model, session, temperature, stream=False, system=None):

    global total_cost

    messages = load_history(session)

    if system:
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": system})

    print(f"\nAI Assistant started! (Session: {session})")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:

        user_input = input("You: ")

        if not user_input.strip():
            continue

        if user_input.lower() in ["quit", "exit"]:
            save_history(messages, session)
            print("Goodbye!")
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            retry_kwargs = {
                "model": model,
                "messages": trim_history(messages),
                "temperature": temperature,
                "stream": stream,
            }
            if stream:
                retry_kwargs["stream_options"] = {"include_usage": True}

            response = call_with_retry(**retry_kwargs)

            if stream:

                collected_content = []
                usage = None

                print("\nAssistant: ", end="", flush=True)

                for chunk in response:

                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        collected_content.append(content)

                    if chunk.usage:
                        usage = chunk.usage

                print("\n")

                assistant_reply = "".join(collected_content)

            else:

                assistant_reply = response.choices[0].message.content
                usage = response.usage

                print("\nAssistant:", assistant_reply, "\n")

            track_cost(usage, model)

        except (AuthenticationError, RateLimitError,
                NotFoundError, APIConnectionError, APIStatusError) as error:

            handle_api_error(error)
            messages.pop()
            continue

        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })

        save_history(messages, session)
        print()

def validate_temperature(value):

    try:
        temperature = float(value)

    except ValueError:
        raise argparse.ArgumentTypeError(
            "Temperature must be a number."
        )

    if not 0.0 <= temperature <= 1.5:
        raise argparse.ArgumentTypeError(
            "Temperature must be between 0.0 and 1.5."
        )

    return temperature

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

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming in interactive mode"
    )

    parser.add_argument(
        "--temperature",
        type=validate_temperature,
        default=0.7,
        help="Control response randomness (0.0 to 1.5)"
    )

    args = parser.parse_args()

    if args.clear:
        clear_history(args.session)

    elif args.structured:
        result = structured_extract(args.structured, model=args.model)

        if result is not None:
            print("\nStructured Output:")
            print(result.model_dump_json(indent=2))

    elif args.question:
        ask_groq(
            args.question,
            args.model,
            temperature=args.temperature,
            system=args.system
        )

    else:
        interactive_chat(
            args.model,
            args.session,
            args.temperature,
            args.stream,
            system=args.system
        )


if __name__ == "__main__":
    main()
