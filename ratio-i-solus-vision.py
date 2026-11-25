# Baseline PCB Design Agent

import asyncio
import json
import os
import re
import base64
import mimetypes
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
MODEL          = "qwen3.5-397b-a17b"

llm             = openai.AsyncOpenAI(base_url=SAIA_BASE_URL, api_key=API_KEY)

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



async def fetch_board_image(session: ClientSession, output_width: int = 2048) -> str:
    result = await session.call_tool("get_board_2d_view", {"layers" : ['F.Cu', 'B.Cu', 'Edge.Cuts', 'F.SilkS', 'B.SilkS']})

    extensions = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }

    for block in result.content:
        text = getattr(block, "text", None)
        if text and text.lstrip().startswith(("<svg", "<?xml")):
            output_path = Path("./board_snapshot.png")
            png_bytes = cairosvg.svg2png(
                bytestring=text.encode("utf-8"), output_width=output_width,
            )
            output_path.write_bytes(png_bytes)
            print(f"[agent] Board image saved to {output_path}")
            return str(output_path)

        data = getattr(block, "data", None)
        mime_type = getattr(block, "mimeType", None)
        if data and mime_type in extensions:
            output_path = Path("./board_snapshot").with_suffix(extensions[mime_type])
            output_path.write_bytes(base64.b64decode(data))
            print(f"[agent] Board image saved to {output_path}")
            return str(output_path)

    raise RuntimeError("get_board_2d_view returned no supported image content")

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

    messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Symbolic Reference Layout from datasheet."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{"image/png"};base64,{base64.b64encode(Path("ti_ref.png").read_bytes()).decode("ascii")}",
                    },
                },
            ],
        })

    nudges = 0

    total_tokens = 0

    for _ in range(max_iter):

        # Get current image
        img = await fetch_board_image(session)
        image_mime = mimetypes.guess_type(img)[0] or "image/png"
        image_data = base64.b64encode(Path(img).read_bytes()).decode("ascii")

        # Remove old images from memory
        messages[:] = [
            message for message in messages
            if not (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), list)
                and message["content"]
                and message["content"][0].get("text") == "Current PCB board view."
            )
        ]

        # Add current image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Current PCB board view."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime};base64,{image_data}",
                    },
                },
            ],
        })

        t = time.time()
        
        print(f"  [agent] → call to {MODEL}")

        response = await llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        elapsed = time.time() - t
        
        print(f"  [agent] → call to {MODEL} took {elapsed:.2f} seconds.")

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
                        "consider the provided reference layout! "
                        "do not change the board outline! "
                        "keep all components on the top side!"
                        "the y-axis is increasing downwards, keep this in mind!"
                        "make sure that in the final result all components are within the board boundary, but do not overlap! "
                        "move the non-fixed components one by one and look at the layout between the steps to accomodate dependencies "
                        "do not try to check all coordinates virtually, but rather use the image for spatial reasoning"
            )

            result = await agent(
                                original_instructions=user_input,
                                session=session,
                                openai_tools=openai_tools
                            )


            
            return result


if __name__ == "__main__":

    path = <PATH> # path to .kicad_pro file

    result = asyncio.run(
                        agent_session(
                            project_path=path,
                        )
                    )


