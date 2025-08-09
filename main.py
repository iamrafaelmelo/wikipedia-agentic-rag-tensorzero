import time
import asyncio
from dotenv import load_dotenv
import wikipedia
from tensorzero import AsyncTensorZeroGateway, ToolCall, ToolResult
from markdownify import markdownify
from rich.console import Console
from rich.prompt import Prompt

# ┌────────────────────────────────────────────────────────────────────────────┐
# │                                   GENERAL                                  │
# └────────────────────────────────────────────────────────────────────────────┘
load_dotenv()
console = Console()
max_inferences = 20

# ┌────────────────────────────────────────────────────────────────────────────┐
# │                                   UTILS                                    │
# └────────────────────────────────────────────────────────────────────────────┘
def stream_tokens_effect(response: str):
    for token in response.split(" "):
        print(token, end=" ", flush=True)
        time.sleep(0.03)
    print()
# ┌────────────────────────────────────────────────────────────────────────────┐
# │                                   TOOLS                                    │
# └────────────────────────────────────────────────────────────────────────────┘
def search_wikipedia(tool_call: ToolCall) -> ToolResult:
    """
    Searches Wikipedia for a given query and returns a list of search results.

    Args:
        tool_call (ToolCall): A tool call object containing the search query in its arguments.
            Expected arguments: {"query": str}
    Returns:
        ToolResult: A tool result containing the newline-separated list of Wikipedia search results.
            The result field contains the search results as a string.
    """
    query = tool_call.arguments["query"]
    search_wikipedia_result = "\n".join(wikipedia.search(query))

    return ToolResult(
        name="search_wikipedia",
        id=tool_call.id,
        result=search_wikipedia_result,
    )

def load_wikipedia_page(tool_call: ToolCall) -> ToolResult:
    """
    Loads and formats the content of a Wikipedia page.

    Args:
        tool_call (ToolCall): A tool call object containing the page title in its arguments.
            Expected arguments: {"title": str}
    Returns:
        ToolResult: A tool result containing the formatted Wikipedia page content.
            The result field contains the page URL and content in Markdown format.
            If the page is not found or there's a disambiguation error, returns an error message.
    """
    try:
        page = wikipedia.page(tool_call.arguments["title"])
        page_markdown = markdownify(page.html()) # to reduce token usage
        load_wikipedia_page_result = (
            f"# URL\n\n{page.url}\n\n# CONTENT\n\n{page_markdown}"
        )
    except wikipedia.exceptions.PageError:
        load_wikipedia_page_result = (
            f"ERROR: page '{tool_call.arguments['title']}' not found."
        )
    except wikipedia.exceptions.DisambiguationError as exception:
        load_wikipedia_page_result = (
            f"ERROR: disambiguation error for '{tool_call.arguments['title']}': {exception}"
        )

    return ToolResult(
        name="load_wikipedia_page",
        id=tool_call.id,
        result=load_wikipedia_page_result,
    )

# ┌────────────────────────────────────────────────────────────────────────────┐
# │                                   AGENTS                                   │
# └────────────────────────────────────────────────────────────────────────────┘
async def wikipedia_agent(question: str) -> str:
    """
    Asks a question to the multi-hop retrieval agent and returns the answer.

    Args:
        question (str): The question to ask the agent.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    Returns:
        str: The answer to the question.
    """
    messages = [
        {
            "role": "user",
            "content": question,
        },
    ]

    episode_id = None
    final_response = None

    async with await AsyncTensorZeroGateway.build_embedded(config_file="./config/tensorzero.toml") as client:
        while True or final_response is None:
            response = await client.inference(
                function_name="wikipedia_agent",
                input={"messages": messages},
                episode_id=episode_id,
            )

            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            episode_id = response.episode_id
            output_content_blocks = []

            if final_response is None:
                continue

            for content_block in response.content:
                if isinstance(content_block, ToolCall):
                    console.log(f"{content_block.name}: {content_block.arguments}")

                    if content_block.name is None or content_block.arguments is None:
                        output_content_blocks.append(
                            ToolResult(
                                name=content_block.raw_name,
                                id=content_block.id,
                                result="ERROR: invalid tool call",
                            )
                        )
                    elif content_block.name == "search_wikipedia":
                        output_content_blocks.append(search_wikipedia(content_block))
                    elif content_block.name == "load_wikipedia_page":
                        output_content_blocks.append(load_wikipedia_page(content_block))
                    elif content_block.name == "think":
                        output_content_blocks.append(
                            ToolResult(
                                name="think",
                                id=content_block.id,
                                result="",
                            )
                        )
                    elif content_block.name == "answer_question":
                        console.log(content_block)
                        final_response = content_block.arguments["answer"]
                        return
                else:
                    console.log(content_block)

            messages.append({
                "role": "user",
                "content": output_content_blocks
            })

    return final_response

# ┌────────────────────────────────────────────────────────────────────────────┐
# │                                     RUN                                    │
# └────────────────────────────────────────────────────────────────────────────┘
async def main():
    while True:
        question = Prompt.ask("Your question")
        response = await wikipedia_agent(question=question)
        stream_tokens_effect(response)

if __name__ == "__main__":
    asyncio.run(main())
