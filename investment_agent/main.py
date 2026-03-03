"""Investment Agent - AI investment analysis agent."""

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, cast

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.yfinance import YFinanceTools
from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global agent instance
agent: Agent | None = None
_initialized = False
_init_lock = asyncio.Lock()


def load_config() -> dict[str, Any]:
    """Load agent config from agent_config.json or return defaults."""
    config_path = Path(__file__).parent / "agent_config.json"

    if config_path.exists():
        try:
            with open(config_path) as f:
                return cast(dict[str, Any], json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️ Failed to load config from {config_path}: {exc}")

    return {
        "name": "investment-agent",
        "description": "AI investment agent that researches stock prices, analyst recommendations, and stock fundamentals using YFinance data.",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
        },
    }


async def initialize_agent() -> None:
    """Initialize the investment agent with proper model and tools."""
    global agent

    # Get API keys from environment
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")

    # Model selection (OpenRouter only)
    if openrouter_api_key:
        model = OpenRouter(
            id=model_name,
            api_key=openrouter_api_key,
            cache_response=True,
            supports_native_structured_outputs=True,
        )
        print(f"✅ Using OpenRouter model: {model_name}")
    else:
        # Define error message separately to avoid TRY003
        error_msg = "No API key provided. Please set OPENROUTER_API_KEY environment variable."
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # Initialize tools
    yfinance_tools = YFinanceTools()

    # Create the investment agent
    agent = Agent(
        name="AI Investment Agent",
        model=model,
        tools=[yfinance_tools],
        description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
        instructions=[
            "Format your response using markdown and use tables to display data where possible.",
            "When comparing stocks, provide detailed analysis including price trends, fundamentals, and analyst recommendations.",
            "Always provide actionable insights for investors.",
        ],
        debug_mode=True,
        markdown=True,
    )
    print("✅ Investment Agent initialized")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages."""
    global agent
    if not agent:
        # Define error message separately to avoid TRY003
        error_msg = "Agent not initialized"
        raise RuntimeError(error_msg)

    # Run the agent and get response
    return await agent.arun(messages)


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages with lazy initialization."""
    global _initialized

    # Lazy initialization on first call
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Investment Agent...")
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def cleanup() -> None:
    """Clean up any resources."""
    print("🧹 Cleaning up Investment Agent resources...")


def main():
    """Run the main entry point for the Investment Agent."""
    parser = argparse.ArgumentParser(description="Bindu Investment Agent")
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "openai/gpt-4o"),
        help="Model ID for OpenRouter (env: MODEL_NAME)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to agent_config.json (optional)",
    )
    args = parser.parse_args()

    # Set environment variables if provided via CLI
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Investment Agent - AI Investment Analysis")
    print("📈 Capabilities: Stock analysis, financial research, investment recommendations")

    # Load configuration
    config = load_config()

    try:
        # Use Bindu server
        print("🚀 Starting Bindu Investment Agent server...")
        print(f"🌐 Server will run on: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3773')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Investment Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup on exit
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
