# Baseline PCB Design Agent

import asyncio
import json
import os
import re
import base64
import openai
import cairosvg
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from datetime import datetime

import creds as creds

# ─────────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────────

# local

# API_KEY         = creds.DGX_API_KEY
# DGX_BASE_URL    = creds.DGX_BASE_URL
# MODEL           = "gemma4:26b"

# llm             = openai.AsyncOpenAI(base_url=DGX_BASE_URL, api_key=API_KEY)

# cloud

API_KEY         = creds.SAIA_API_KEY
SAIA_BASE_URL   = creds.SAIA_BASE_URL
#MODEL           = "mistral-medium-3.5-128b" 
#MODEL           = "glm-5.3-flash" 
#MODEL           = "qwen3.6-35b-a3b"
#MODEL           = "qwen3-coder-next"
#MODEL           = "devstral-2-123b-instruct-2512"
MODEL          = "qwen3.5-397b-a17b"

llm             = openai.AsyncOpenAI(base_url=SAIA_BASE_URL, api_key=API_KEY)

RATE_LIMIT_SLEEP_S     = 3600   # 1 hour
MAX_RATE_LIMIT_RETRIES = 6      # give up after ~6 h

# Total time spent waiting, so you can subtract it from the runtime metric
sleep_total = 0.0


async def llm_call_with_retry(role: str, **kwargs):
    """chat.completions.create with a 1 h sleep whenever the token/rate quota is exhausted."""
    global sleep_total
    attempt = 0
    while True:
        try:
            return await llm.chat.completions.create(**kwargs)
        except (openai.RateLimitError, openai.APIStatusError) as e:
            status = getattr(e, "status_code", None)
            # 429 = rate limit / quota. 402 is used by some providers for "out of credits".
            if status not in (402, 429):
                raise
            attempt += 1
            if attempt > MAX_RATE_LIMIT_RETRIES:
                print(f"  [{role}] quota still exhausted after {attempt - 1} retries, giving up.")
                raise
            print(f"  [{role}] quota/rate limit hit ({status}): {e}")
            print(f"  [{role}] sleeping {RATE_LIMIT_SLEEP_S // 60} min "
                  f"(retry {attempt}/{MAX_RATE_LIMIT_RETRIES}) ...")
            await asyncio.sleep(RATE_LIMIT_SLEEP_S)
            sleep_total += RATE_LIMIT_SLEEP_S

# ─────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────

def mcp_tool_to_openai(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }

def save_completion_to_md(response, filepath="output.md"):
    choice = response.choices[0]
    message = choice.message

    # --- Parse the main content (strip ```json fences if present) ---
    raw_content = message.content or ""

    lines = []
    lines.append(f"# Chat Completion Result\n")
    lines.append(f"**Model:** `{response.model}`  ")
    lines.append(f"**Created:** {datetime.fromtimestamp(response.created)}  ")
    lines.append(f"**Finish reason:** `{choice.finish_reason}`\n")


    lines.append("## Raw Content\n")
    lines.append("```json")
    lines.append(raw_content)
    lines.append("```\n")

    # --- Reasoning (chain-of-thought), collapsible ---
    reasoning = getattr(message, "reasoning", None)

    if reasoning:
        lines.append("## Reasoning\n")
        lines.append("<details>")
        lines.append("<summary>Click to expand model reasoning</summary>\n")
        lines.append(reasoning)
        lines.append("\n</details>\n")

    # --- Usage stats ---
    if response.usage:
        lines.append("## Usage\n")
        lines.append(f"- Prompt tokens: {response.usage.prompt_tokens}")
        lines.append(f"- Completion tokens: {response.usage.completion_tokens}")
        lines.append(f"- Total tokens: {response.usage.total_tokens}\n")

    Path(filepath).write_text("\n".join(lines), encoding="utf-8")

    print(f"[orchestrator] Trace saved to {filepath}")


# ─────────────────────────────────────────────
# Agent Definition
# ─────────────────────────────────────────────
FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Call this ONLY when the task is fully completed and verified. "
            "Provide the final answer/summary for the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Final answer / summary of what was done.",
                }
            },
            "required": ["summary"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are a PCB design assistant. Solve the user's query with the available tools. "
    "Use the save_project tool when you completed your placement. "
    "When (and only when) the task is completely done, call the `finish` tool with a final summary."
)

NUDGE = (
    "You replied without calling a tool. Do not just describe plans. "
    "Continue the task by calling the next tool, or call `finish` if the task is complete."
)


async def agent(original_instructions: str,
                session: ClientSession,
                openai_tools: list[dict],
                max_iter: int = 50,
                max_nudges: int = 5,
            ) -> tuple[bool, str, int]:
    
    tools = openai_tools

    tool_kit        = ["save_project"
                     , "close_project"
                     , "save_board"
                     , "get_project_info"
                     , "get_board_origin"
                     , "get_board_info"
                     , "get_board_extents"
                     , "get_board_2d_view"
                     , "move_component"
                     , "rotate_component"
                     , "find_component"
                     , "get_component_properties"
                     , "get_component_pads"
                     , "get_pads"
                     , "get_net_pads"
                     , "get_component_geometry"
                     , "get_component_list"
                     , "get_pad_position"
                     , "get_ratsnest"
                     , "check_placement_clearance"
                     , "check_courtyard_overlaps"
                     , "run_drc"
                     , "check_clearance"
                     , "get_drc_violations"
                     , "export_svg"
                     , "list_schematic_components"
                     , "list_schematic_nets"
                    ]

    tools = [tool for tool in tools if tool.get("function").get("name") in tool_kit] + [FINISH_TOOL]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User Query: {original_instructions}"},
    ]

    saved_image_path = None
    nudges = 0

    total_tokens = 0

    for _ in range(max_iter):

        t = time.time()

        print(f"  [agent] → call to {MODEL}")

        response = await llm_call_with_retry("agent", model=MODEL, messages=messages, tools=tools)

        elapsed = time.time() - t

        print(f"  [agent] → call to {MODEL} took {elapsed:.2f} seconds.)")

        total_tokens = total_tokens + response.usage.total_tokens

        now = datetime.now()
        day = now.strftime("%m_%d")
        os.makedirs(f"./traces/{day}", exist_ok=True)
        trace_path = f"./traces/{day}/trace_{now.strftime('%H_%M_%S')}.md"
        save_completion_to_md(response=response, filepath=trace_path)

        msg = response.choices[0].message
        messages.append(msg)

        # --- No tool call: model just talked. Nudge instead of terminating. ---
        if not msg.tool_calls:
            nudges += 1
            if nudges > max_nudges:
                return False, msg.content or "Agent stopped calling tools without finishing."
            messages.append({"role": "user", "content": NUDGE})
            continue

        nudges = 0  # reset once the model behaves

        # --- Dispatch tool calls ---
        for tc in msg.tool_calls:
            fn = tc.function.name

            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Error: tool arguments were not valid JSON. Retry.",
                })
                continue

            print(f"  [agent] → {fn}({args})")

            # Explicit termination
            if fn == "finish":
                drc_result      = await session.call_tool("get_drc_violations", {"severity" : "error"})
                drc_text        = json.loads(drc_result.content[0].text) if drc_result.content else "{}"
                v               = len(drc_text["violations"])
                return True, args.get("summary", ""), total_tokens, v

            try:
                result = await session.call_tool(fn, args)
            except Exception as e:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Tool error: {e}",
                })
                continue

            result_text = result.content[0].text if result.content else "ok"

            # Intercept board images
            if fn in ("get_board_2d_view", "export_svg", "get_schematic_view"):
                found = False
                for i, block in enumerate(result.content):
                    if getattr(block, "text", None):
                        text = block.text.strip()
                        if text.startswith("<svg") or text.startswith("<?xml"):
                            path = "./board_snapshot.svg"
                            with open(path, "w") as f:
                                f.write(block.text)
                            saved_image_path = path
                            result_text = f"Board image saved to {path}"
                            found = True
                            break
                    elif getattr(block, "data", None):
                        path = "./board_snapshot.png"
                        with open(path, "wb") as f:
                            f.write(base64.b64decode(block.data))
                        saved_image_path = path
                        result_text = f"Board image saved to {path}"
                        found = True
                        break
                if not found:
                    print(f"  [agent] ✗ no image found in {fn} result")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })


    drc_result      = await session.call_tool("get_drc_violations", {"severity" : "error"})
    drc_text        = json.loads(drc_result.content[0].text) if drc_result.content else "{}"
    v               = len(drc_text["violations"])

    return False, "Max iterations reached without calling finish.", total_tokens, v


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────
async def agent_session(project_path: str):

    server_params = StdioServerParameters(
        command="podman",
        args=[
            "run", "--rm", "-i",
            "-v", f"{os.environ['HOME']}/kicad_workspace:/workspace/projects:Z",
            "kicad-mcp",
            "node", "/workspace/KiCAD-MCP-Server/dist/index.js"
        ],
        env={},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            await session.initialize()
            mcp_tools = await session.list_tools()
            openai_tools = [mcp_tool_to_openai(t) for t in mcp_tools.tools]

            print(f"[orchestrator] {len(mcp_tools.tools)} tools available")
            print("[orchestrator] type 'exit' to quit\n")

            # Open the project (which initializes the board object)
            print("[orchestrator] opening the project...")
            await session.call_tool("open_project", {"filename": project_path})
            print("[orchestrator] project is loaded.")

            # # Interactive mode

            # while True:

            #     try:
            #         user_input = await asyncio.get_event_loop().run_in_executor(
            #             None, lambda: input("You: ").strip()
            #         )
            #     except (EOFError, KeyboardInterrupt):
            #         print("\n[orchestrator] session ended.")
            #         break

            #     if user_input.lower() in ("exit", "quit", ""):
            #         break


            #     result = await agent(
            #                         original_instructions=user_input,
            #                         session=session,
            #                         openai_tools=openai_tools
            #                     )

            #     print(f"[RATIO]: {result}")

            # # Automated mode

            user_input =(
                        "execute a placement for the dcdc converter project to which you are connected via mcp. " 
                        "the placement for j1, j2 and u1 is fixed, so do not move them! arrange the other components in a way a power electronics pcb design expert would do. "
                        "do not change the board outline! "
                        "make sure that in the final result all components are within the board boundary, but do not overlap! "
            )

            result = await agent(
                                original_instructions=user_input,
                                session=session,
                                openai_tools=openai_tools
                            )


            
            return result


if __name__ == "__main__":

    for i in range(10):

        path = <<PATH>>

        result = asyncio.run(
                            agent_session(
                                project_path=path,
                            )
                        )


