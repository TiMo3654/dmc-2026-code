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

import creds

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


def svg_to_png_base64(svg_path: str) -> str:
    png_bytes = cairosvg.svg2png(url=svg_path)
    return base64.b64encode(png_bytes).decode("utf-8")


# ─────────────────────────────────────────────
# Context retrieval
# ─────────────────────────────────────────────

async def fetch_schematic_context(session: ClientSession, project_path: str) -> str:
    """
    Queries the schematic for component functions, net topology,
    and special connections to inform placement decisions.
    """
    lines = ["=== SCHEMATIC CONTEXT ==="]

    # ── Component list with values and descriptions ──
    result = await session.call_tool("list_schematic_components", {"schematicPath" : project_path.replace("_pro", "_sch")})

    components = result.content[0].text.splitlines()

    try:
        lines.append(f" Textual schematic description: {components}")
    except Exception as e:
        lines.append(f"  [could not parse component list: {e}]")

    # ── Net list — reveals what connects to what ──
    result = await session.call_tool("list_schematic_nets", {"schematicPath" : project_path.replace("_pro", "_sch")})

    try:
        lines.append(result.content[0].text)

    except Exception as e:
        lines.append(f"  [could not parse net list: {e}]")

    return "\n".join(lines)


async def fetch_board_context(session: ClientSession) -> str:
    """
    Opens the board and queries get_board_extents + get_component_list
    to build a dynamic spatial context string.
    """

    # Query board extents
    ext_result = await session.call_tool("get_board_extents", {})
    ext_text = ext_result.content[0].text if ext_result.content else "{}"
    try:
        ext = json.loads(ext_text)
    except json.JSONDecodeError:
        print(f"[context] could not parse get_board_extents: {ext_text}")
        ext = {}

    # Query drc
    drc_result = await session.call_tool("get_drc_violations", {"severity" : "error"})
    drc_text = drc_result.content[0].text if drc_result.content else "{}"
    try:
        drc = json.loads(drc_text)
        drc_report = ["=== DRC Report ==="]
        for vio in drc['violations']:
            drc_report.append(f"Violation type: {vio["type"]}, Severity: {vio["severity"]}, Message: {vio["message"]}, Location: x : {vio["location"]["x"]}; y : {vio["location"]["y"]} ")
    except json.JSONDecodeError:
        print(f"[context] could not parse get_drc_violations: {drc_text}")
        drc = {}

    # Extract boundary values — handle both possible key naming conventions
    bx1 = float(ext['extents'].get("left"))
    bx2 = float(ext['extents'].get("right"))
    by1  = float(ext['extents'].get("top"))
    by2 = float(ext['extents'].get("bottom"))
    bw = float(ext['extents'].get("width"))
    bh = float(ext['extents'].get("height"))

    cx  = float(ext['extents']['center'].get("x"))
    cy  = float(ext['extents']['center'].get("y"))

    # Query component list for positions
    comp_result = await session.call_tool("get_component_list", {})
    comp_text = comp_result.content[0].text if comp_result.content else "[]"
    try:
        components = json.loads(comp_text)['components']
    except json.JSONDecodeError:
        components = []

    # Build context string
    lines = [
        "=== LIVE BOARD CONTEXT ===",
        f"Board boundary: ({bx1:.2f}, {by1:.2f})mm  →  ({bx2:.2f}, {by2:.2f})mm",
        f"Board size:     {bw:.2f}mm wide  x  {bh:.2f}mm tall",
        f"Board centre:   ({cx:.2f}, {cy:.2f})mm",
        f"",
        f"Placement constraints:",
        f"  Valid X range: {bx1 + 2:.2f}mm  to  {bx2 - 2:.2f}mm  (2mm edge clearance)",
        f"  Valid Y range: {by1 + 2:.2f}mm  to  {by2 - 2:.2f}mm  (2mm edge clearance)",
        f"  West edge zone (connectors): X < {bx1 + 12:.2f}mm",
        f"  East edge zone (connectors): X > {bx2 - 12:.2f}mm",
        f"  North edge zone:             Y < {by1 + 12:.2f}mm",
        f"  South edge zone:             Y > {by2 - 12:.2f}mm",
        f"",
        f"Component positions ({len(components)} total):",
    ]

    for comp in components:
        ref = comp.get("reference", "?")
        x   = float(comp['position'].get("x", 0))
        y   = float(comp['position'].get("y", 0))
        rot = comp.get("rotation", 0)

        # Describe position relative to board
        rel_x = "left half" if x < cx else "right half"
        rel_y = "top half"  if y < cy else "bottom half"

        # Edge proximity
        near = []
        if x - bx1 < 15: near.append(f"near west edge ({x - bx1:.1f}mm)")
        if bx2 - x < 15: near.append(f"near east edge ({bx2 - x:.1f}mm)")
        if y - by1 < 15: near.append(f"near north edge ({y - by1:.1f}mm)")
        if by2 - y < 15: near.append(f"near south edge ({by2 - y:.1f}mm)")

        # Out of bounds flag
        oob = " *** OUT OF BOUNDS ***" if (
            x < bx1 or x > bx2 or y < by1 or y > by2
        ) else ""

        pos_desc = ", ".join(near) if near else f"{rel_x}, {rel_y}"
        lines.append(
            f"  {ref:8s}: ({x:7.2f}, {y:7.2f})mm  rot={rot}°  — {pos_desc}{oob}"
        )

        # Pad information

        lines.append(f"     COMPONENT {ref} - PAD INFORMATION")

        cpads = await session.call_tool("get_component_pads", {"reference" : ref})

        pads_text = cpads.content[0].text if cpads.content else "[]"
        try:
            pads = json.loads(pads_text)['pads']
        except json.JSONDecodeError:
            print(f"[context] could not parse get_component_pads: {cpads}")
            pads = []

        #print(pads)

        for pad in pads:
            n   = pad.get("number")
            x   = float(pad['position'].get("x", 0))
            y   = float(pad['position'].get("y", 0))
            net = pad.get("net")
            
            lines.append(
                f"Component {ref} - Pad Number: {n} - Pad Net {net} - Pad X Position {x} - Pad Y Position {y}"
            )

    # DRC Report    
    
    lines += drc_report

    return "\n".join(lines)


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
                project_path: str,
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

    print("[orchestrator] fetching live board context...")
    schematic_context = await fetch_schematic_context(session, project_path)

    print(schematic_context)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User Query: {original_instructions}"},
        {"role": "user", "content": f"Schematic Context: {schematic_context}"},
    ]

    saved_image_path = None
    nudges = 0

    total_tokens = 0
   

    for i in range(max_iter):

        print("[orchestrator] fetching live board context...")
        board_context = await fetch_board_context(session)

        if i < 3:
            print(board_context)

        messages.append({"role": "user", "content" : f"Live Board Context: {board_context}"})

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
                        "keep all components on the top side!"
                        "the y-axis is increasing downwards, keep this in mind!"
                        "make sure that in the final result all components are within the board boundary, but do not overlap! "
                        "move the non-fixed components one by one and look at the component coordinates between the steps to accomodate dependencies. "
                        "here are the layout guidelines provided by the ic (u1) datasheet: "
                        "1. vin and gnd traces should be as wide as possible to reduce trace impedance. the wide areas are also of advantage from the view point of heat dissipation. "
                        "2. the input capacitor and output capacitor should be placed as close to the device as possible to minimize trace impedance. "
                        "3. provide sufficient vias for the input capacitor and output capacitor. "
                        "4. keep the sw trace as physically short and wide as practical to minimize radiated emissions. "
                        "5. do not allow switching current to flow under the device. "
                        "6. a separate vout path should be connected to the upper feedback resistor. "
                        "7. make a kelvin connection to the gnd pin for the feedback path. "
                        "8. voltage feedback loop should be placed away from the high-voltage switching trace, and preferably has ground shield. "
                        "9. the trace of the fb node should be as small as possible to avoid noise coupling. "
                        "10. the gnd trace between the output capacitor and the gnd pin should be as wide as possible to minimize its trace impedance. "
                        "consider these guidelines for your placement proposal!"
            )

            result = await agent(
                                original_instructions=user_input,
                                session=session,
                                project_path=project_path,
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

