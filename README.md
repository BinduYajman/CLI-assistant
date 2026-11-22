# CLI AI Assistant

A resilient, feature-packed Command Line AI Assistant powered by Groq and Pydantic.

## Architecture

```mermaid
graph TD
    A[User CLI Input] --> B[Argument Parser]
    B --> C{Operation}

    C -->|question| D[One-shot Chat]
    C -->|default| E[Interactive Chat]
    C -->|--structured| F[Structured Extraction]
    C -->|--clear| N[Clear History]

    E <--> H[(Session History JSON)]
    E --> O[Trim History Window]

    D --> P[Retry Middleware: call_with_retry]
    O --> P
    F --> P

    P <--> G[Groq API]

    D -->|stream=True| I[Streaming Output]
    E -->|--stream| I
    E -->|default: stream=False| Q[Direct Text Output]

    I --> J[Token & Cost Tracking]
    Q --> J
    J <--> R[Dynamic Pricing Scraper / Fallback Cache]

    G -->|JSON Schema Response| S[JSON Parse & Pydantic Validation]
    S --> M[Structured JSON Output]
    S --> J
```

## Features

- **One-shot Questions**: Ask quick questions directly from the command line with streaming output.
- **Interactive Chat**: Multi-turn chat with persistent session history and configurable rolling context windows.
- **Session Management**: Maintain isolated conversation histories using `--session <name>`.
- **Structured Data Extraction**: Extract validated JSON with strict Pydantic schemas.
- **Model Switching & Temperature Control**: Customize model (`--model`) and sampling randomness (`--temperature`).
- **Resilient API Calls**: Automatic exponential backoff retry mechanism on rate limits, connection errors, and 5xx server issues.
- **Token & Cost Tracking**: Live cost estimation with dynamic pricing scraping and reliable local fallbacks.
