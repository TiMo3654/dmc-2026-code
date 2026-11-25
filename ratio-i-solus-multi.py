# Multi-Agent PCB Design System: Supervisor / Executor / Reviewer
#
#   Supervisor : analyzes schematic + reference layout, plans coordinates, decides when to
#                execute, when to ask for a review, and when it is satisfied (finish).
#   Executor   : blindly executes the coordinate plan handed over by the supervisor.
#   Reviewer   : read-only critic. Compares current layout with the reference layout and
#                returns structured feedback to the supervisor.
#
# The supervisor is the only "long-lived" agent (its message history persists over all rounds).
# Executor and reviewer are spawned fresh for every call and exposed to the supervisor as tools:
#   execute_placement(placements)  -> runs the executor
#   request_review(focus)          -> runs the reviewer
#   finish(summary)                -> ends the run

import asyncio
import base64
import itertools
import json
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

import cairosvg
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import creds as creds

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

API_KEY       = creds.SAIA_API_KEY
SAIA_BASE_URL = creds.SAIA_BASE_URL
MODEL         = "qwen3.5-397b-a17b"
#MODEL           = "qwen3.6-35b-a3b"
#MODEL           = "mistral-medium-3.5-128b" 


# One model per role (the reviewer needs vision; the executor could be a cheaper model)
MODELS = {
    "supervisor": MODEL,
    "executor":   MODEL,
    "reviewer":   MODEL,
}

llm = openai.AsyncOpenAI(base_url=SAIA_BASE_URL, api_key=API_KEY)

REFERENCE_IMAGE = "ti_ref.png"

# Components that must never be moved (enforced in code, not only in the prompt)
FIXED_COMPONENTS = {"J1", "J2", "U1"}

# Budgets
MAX_ROUNDS            = 25     # max. executor invocations (plan -> execute cycles)
MAX_REVIEWS           = 5
SUPERVISOR_MAX_ITER   = 90    # 10 + 3*25 + 5 = 90 minimum, plus slack for retries/nudges
EXECUTOR_MAX_ITER     = 25    # 6 movable parts: ~6 moves + 6 rotations + verify + save + report
REVIEWER_MAX_ITER     = 12

# Tool kits per role (names of the KiCAD MCP tools)
SUPERVISOR_TOOLKIT = [
    "get_project_info", "get_board_origin", "get_board_info", "get_board_extents",
    "find_component", "get_component_properties", "get_component_pads", "get_pads",
    "get_net_pads", "get_component_geometry", "get_component_list", "get_pad_position",
    "get_ratsnest", "list_schematic_components", "list_schematic_nets",
]

EXECUTOR_TOOLKIT = [
    "move_component", "rotate_component", "get_component_properties",
    "get_component_geometry", "get_component_list", "get_board_extents",
    "check_placement_clearance", "check_courtyard_overlaps",
    "save_project", "save_board",
]

REVIEWER_TOOLKIT = [
    "get_component_list", "get_component_geometry", "get_component_pads", "get_net_pads",
    "get_board_extents", "get_ratsnest", "check_courtyard_overlaps",
    "check_placement_clearance", "check_clearance", "run_drc", "get_drc_violations",
    "list_schematic_nets",
]

# NOTE: get_board_2d_view is deliberately not in any toolkit: the board image is injected
# automatically into every vision-enabled agent before each LLM call.

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
# Shared helpers
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


def make_tool(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": parameters}}


def select_tools(all_tools: list[dict], names: list[str]) -> list[dict]:
    return [t for t in all_tools if t["function"]["name"] in names]


def image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def image_message(label: str, path: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": label},
            {"type": "image_url", "image_url": {"url": image_to_data_url(path)}},
        ],
    }


def tool_result_to_text(result) -> str:
    """Robustly flatten an MCP tool result (text blocks only; images are not forwarded)."""
    parts = [b.text for b in (result.content or []) if getattr(b, "text", None)]
    if parts:
        return "\n".join(parts)
    return "ok" if not result.content else "[non-text tool result omitted]"


@dataclass
class Usage:
    total: int = 0
    by_role: dict = field(default_factory=dict)

    def add(self, role: str, tokens: int):
        self.total += tokens
        self.by_role[role] = self.by_role.get(role, 0) + tokens


_trace_counter = itertools.count()


def save_completion_to_md(response, role: str):
    now = datetime.now()
    folder = f"./traces/{now.strftime('%m_%d')}"
    os.makedirs(folder, exist_ok=True)
    filepath = f"{folder}/trace_{now.strftime('%H_%M_%S')}_{next(_trace_counter):03d}_{role}.md"

    choice = response.choices[0]
    message = choice.message

    lines = [
        f"# Chat Completion Result ({role})\n",
        f"**Model:** `{response.model}`  ",
        f"**Created:** {datetime.fromtimestamp(response.created)}  ",
        f"**Finish reason:** `{choice.finish_reason}`\n",
        "## Raw Content\n",
        "```json",
        message.content or "",
        "```\n",
    ]

    if message.tool_calls:
        lines.append("## Tool calls\n")
        for tc in message.tool_calls:
            lines.append(f"- `{tc.function.name}({tc.function.arguments})`")
        lines.append("")

    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        lines += ["## Reasoning\n", "<details>",
                  "<summary>Click to expand model reasoning</summary>\n",
                  reasoning, "\n</details>\n"]

    if response.usage:
        lines += ["## Usage\n",
                  f"- Prompt tokens: {response.usage.prompt_tokens}",
                  f"- Completion tokens: {response.usage.completion_tokens}",
                  f"- Total tokens: {response.usage.total_tokens}\n"]

    Path(filepath).write_text("\n".join(lines), encoding="utf-8")


async def fetch_board_image(session: ClientSession, output_width: int = 2048) -> str:
    result = await session.call_tool(
        "get_board_2d_view",
        {"layers": ["F.Cu", "B.Cu", "Edge.Cuts", "F.SilkS", "B.SilkS"]},
    )

    extensions = {"image/png": ".png", "image/jpeg": ".jpg",
                  "image/webp": ".webp", "image/gif": ".gif"}

    for block in result.content:
        text = getattr(block, "text", None)
        if text and text.lstrip().startswith(("<svg", "<?xml")):
            output_path = Path("./board_snapshot.png")
            png_bytes = cairosvg.svg2png(bytestring=text.encode("utf-8"),
                                         output_width=output_width)
            output_path.write_bytes(png_bytes)
            return str(output_path)

        data = getattr(block, "data", None)
        mime_type = getattr(block, "mimeType", None)
        if data and mime_type in extensions:
            output_path = Path("./board_snapshot").with_suffix(extensions[mime_type])
            output_path.write_bytes(base64.b64decode(data))
            return str(output_path)

    raise RuntimeError("get_board_2d_view returned no supported image content")


BOARD_VIEW_LABEL = "Current PCB board view."


async def refresh_board_message(session: ClientSession, messages: list):
    """Drop stale board snapshots from the history and append the current one."""
    img = await fetch_board_image(session)
    messages[:] = [
        m for m in messages
        if not (isinstance(m, dict)
                and m.get("role") == "user"
                and isinstance(m.get("content"), list)
                and m["content"]
                and m["content"][0].get("text") == BOARD_VIEW_LABEL)
    ]
    messages.append(image_message(BOARD_VIEW_LABEL, img))


async def count_drc_errors(session: ClientSession) -> int:
    try:
        res = await session.call_tool("get_drc_violations", {"severity": "error"})
        data = json.loads(res.content[0].text) if res.content else {}
        return len(data.get("violations", []))
    except Exception as e:
        print(f"[orchestrator] DRC count failed: {e}")
        return -1


# ─────────────────────────────────────────────
# Generic agent loop (used by all three roles)
# ─────────────────────────────────────────────

ToolHandler = Callable[[dict], Awaitable[str]]

NUDGE = (
    "You replied without calling a tool. Do not just describe plans. "
    "Continue the task by calling the next tool."
)


@dataclass
class LoopResult:
    terminal_tool: Optional[str]      # name of the terminal tool that ended the loop (None = no)
    terminal_args: dict
    last_text: str = ""


async def run_agent_loop(
    *,
    role: str,
    messages: list,
    tools: list[dict],
    session: ClientSession,
    usage: Usage,
    terminal_tools: set[str],
    handlers: Optional[dict[str, ToolHandler]] = None,
    max_iter: int = 50,
    max_nudges: int = 5,
    show_board: bool = True,
) -> LoopResult:
    """
    Runs an LLM <-> tool loop on `messages` (mutated in place).
    - Calls to a tool in `terminal_tools` end the loop and are returned to the caller.
    - Calls to a tool in `handlers` are routed to a Python coroutine (e.g. sub-agents).
    - Everything else is forwarded to the MCP server.
    """
    handlers = handlers or {}
    model = MODELS[role]
    nudges = 0

    for _ in range(max_iter):
        if show_board:
            await refresh_board_message(session, messages)

        t0 = time.time()
        print(f"  [{role}] → call to {model}")
        response = await llm_call_with_retry(role, model=model, messages=messages, tools=tools)
        print(f"  [{role}] → took {time.time() - t0:.2f}s")

        if response.usage:
            usage.add(role, response.usage.total_tokens)
        save_completion_to_md(response, role)

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            nudges += 1
            if nudges > max_nudges:
                return LoopResult(None, {}, msg.content or "Agent stopped calling tools.")
            messages.append({"role": "user", "content": NUDGE})
            continue
        nudges = 0

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": "Error: tool arguments were not valid JSON. Retry."})
                continue

            print(f"  [{role}] → {name}({json.dumps(args)[:300]})")

            if name in terminal_tools:
                return LoopResult(name, args)

            try:
                if name in handlers:
                    out = await handlers[name](args)
                else:
                    out = tool_result_to_text(await session.call_tool(name, args))
            except Exception as e:
                out = f"Tool error: {e}"

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})

    return LoopResult(None, {}, "Max iterations reached.")


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

FIXED_STR = ", ".join(sorted(FIXED_COMPONENTS))

COMMON_RULES = (
    f"Fixed components that must NEVER be moved: {FIXED_STR}. "
    "Do not change the board outline. Keep all components on the top side. "
    "Coordinates are in mm in the KiCad board coordinate system; the y-axis increases DOWNWARDS "
    "(a larger y means further down in the board image). "
    "All components must end up inside the board boundary without overlapping courtyards."
)

SUPERVISOR_PROMPT = f"""You are the SUPERVISOR of a three-agent PCB placement team (supervisor, executor, reviewer).
You do not move components yourself. You analyze, plan, delegate and decide.

{COMMON_RULES}

Your team:
- EXECUTOR (tool `execute_placement`): moves/rotates components to exactly the coordinates you give.
  It does no design thinking, so every placement needs explicit x, y (and rotation if needed).
- REVIEWER (tool `request_review`): read-only critic that compares the current layout with the
  reference layout and returns structured feedback. YOU decide when it is worth calling
  (e.g. after the first full placement, after a bigger rework, or before finishing). You are not
  forced to follow its advice, but you must consider it.

Recommended workflow:
1. ANALYZE: read the schematic (list_schematic_components, list_schematic_nets), board info and
   extents, component list/geometry (footprint sizes!), pads of the IC and the ratsnest.
2. REASON about constraints as a power electronics layout expert: identify the switching/hot loop
   and keep it small, input capacitor close to the IC supply/ground pins, inductor close to the
   switch node pin with a compact switch node, output capacitor near the inductor/output, feedback
   divider close to the FB pin and kept away from the switch node, short high-current paths,
   sensible orientation, leave room for routing and keep clearances.
3. Use the reference layout image to derive the RELATIVE arrangement (it is symbolic, not to scale).
   Translate it into absolute board coordinates using the fixed components and the board extents.
4. PLAN concrete coordinates for every non-fixed component and call `execute_placement`.
   You may place in stages (most critical components first) if that helps.
5. LOOK at the updated board image (it is attached automatically). Use the image for spatial
   reasoning rather than computing everything numerically. Correct what you see.
6. Optionally call `request_review`, then refine with another `execute_placement`.
7. When you are satisfied (everything inside the board, no overlaps, layout resembles the
   reference and respects the constraints), call `finish`.

Budget: at most {MAX_ROUNDS} executor rounds and {MAX_REVIEWS} reviews. When a budget is exhausted
the corresponding tool refuses to run, so plan your rounds wisely and call `finish` in time.
Only components listed in a placement are moved; all others stay where they are."""

EXECUTOR_PROMPT = f"""You are the EXECUTOR of a PCB placement team. A supervisor has already done all
design reasoning. Your ONLY job is to apply the given placement plan exactly.

{COMMON_RULES}

Rules:
- For every entry call move_component with exactly the given x/y and, if a rotation is given,
  rotate_component. Do NOT alter coordinates, add, skip or reorder components on your own.
- Never move {FIXED_STR}.
- After the moves, verify each moved component (e.g. get_component_properties) and optionally run
  check_courtyard_overlaps / check_placement_clearance. Only REPORT problems; do not fix them
  by improvising new coordinates.
- Call save_project once all moves are done.
- Finish by calling `report_done` with a short summary and a list of any failures or problems."""

REVIEWER_PROMPT = f"""You are the REVIEWER of a PCB placement team. You are read-only: you never move
anything. You receive the current board image, the symbolic reference layout from the datasheet, and
the supervisor's context. Critically assess the current placement and give actionable feedback.

{COMMON_RULES}

Check at least:
- Hard constraints: components inside the board, no overlapping courtyards/pads (use
  check_courtyard_overlaps, check_placement_clearance, get_board_extents, run_drc).
- Similarity of the RELATIVE arrangement to the reference layout.
- Power electronics quality: size of the hot loop, input/output capacitor proximity, inductor and
  switch node compactness, feedback divider placement and distance to the switch node, ratsnest
  crossings (get_ratsnest), orientation, routing space.
Suggestions should be concrete but relative (e.g. "move C1 ~2 mm left and closer to U1 VIN pad"),
since the supervisor owns the final coordinates. Remember: y increases downwards.
Finish by calling `submit_review`. Use verdict "approve" only if you see no critical or major issue."""


# ─────────────────────────────────────────────
# Control tools
# ─────────────────────────────────────────────

EXECUTE_PLACEMENT_TOOL = make_tool(
    "execute_placement",
    "Delegate a placement to the executor. It moves (and rotates) exactly the listed components to "
    "the given coordinates. Components not listed stay where they are. Returns the executor report.",
    {
        "type": "object",
        "properties": {
            "placements": {
                "type": "array",
                "description": "Target poses of the components to move.",
                "items": {
                    "type": "object",
                    "properties": {
                        "reference": {"type": "string", "description": "Designator, e.g. 'C1'."},
                        "x": {"type": "number", "description": "Target x in mm."},
                        "y": {"type": "number", "description": "Target y in mm (increases downwards)."},
                        "rotation": {"type": "number", "description": "Target rotation in degrees (optional)."},
                    },
                    "required": ["reference", "x", "y"],
                },
            },
            "rationale": {"type": "string",
                          "description": "Short explanation of the intent of this step."},
        },
        "required": ["placements"],
    },
)

REQUEST_REVIEW_TOOL = make_tool(
    "request_review",
    "Ask the reviewer to assess the current layout against the reference layout. "
    "Returns structured feedback (verdict, issues, suggestions).",
    {
        "type": "object",
        "properties": {
            "focus": {"type": "string",
                      "description": "Optional: what the reviewer should pay special attention to."},
        },
    },
)

SUPERVISOR_FINISH_TOOL = make_tool(
    "finish",
    "Call ONLY when you are satisfied with the layout. Provide the final summary.",
    {
        "type": "object",
        "properties": {"summary": {"type": "string",
                                   "description": "Final summary of the placement and its rationale."}},
        "required": ["summary"],
    },
)

EXECUTOR_DONE_TOOL = make_tool(
    "report_done",
    "Call when the plan has been executed. Report the outcome to the supervisor.",
    {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "What was done."},
            "failures": {"type": "array", "items": {"type": "string"},
                         "description": "Problems encountered (failed moves, overlaps, out of bounds). Empty if none."},
        },
        "required": ["summary"],
    },
)

SUBMIT_REVIEW_TOOL = make_tool(
    "submit_review",
    "Submit your review to the supervisor.",
    {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approve", "revise"]},
            "score": {"type": "integer", "minimum": 1, "maximum": 10,
                      "description": "Overall layout quality."},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "components": {"type": "array", "items": {"type": "string"}},
                        "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                        "problem": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["components", "severity", "problem", "suggestion"],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["verdict", "score", "issues", "summary"],
    },
)


def format_review(r: dict) -> str:
    lines = [f"REVIEW VERDICT: {r.get('verdict')}  (score {r.get('score')}/10)",
             f"Summary: {r.get('summary', '')}"]
    for i, issue in enumerate(r.get("issues", []), 1):
        lines.append(f"{i}. [{issue.get('severity')}] {', '.join(issue.get('components', []))}: "
                     f"{issue.get('problem')}\n   Suggestion: {issue.get('suggestion')}")
    if not r.get("issues"):
        lines.append("No issues reported.")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Sub-agents
# ─────────────────────────────────────────────

async def run_executor(session, all_tools, usage, placements: list[dict], rationale: str) -> str:
    tools = select_tools(all_tools, EXECUTOR_TOOLKIT) + [EXECUTOR_DONE_TOOL]
    plan_text = json.dumps(placements, indent=2)
    messages = [
        {"role": "system", "content": EXECUTOR_PROMPT},
        {"role": "user", "content": (
            f"Execute this placement plan exactly.\nIntent (for context only): {rationale or 'n/a'}\n\n"
            f"Plan:\n{plan_text}")},
    ]
    res = await run_agent_loop(
        role="executor", messages=messages, tools=tools, session=session, usage=usage,
        terminal_tools={"report_done"}, max_iter=EXECUTOR_MAX_ITER, show_board=False,
    )
    if res.terminal_tool == "report_done":
        failures = res.terminal_args.get("failures") or []
        report = f"EXECUTOR REPORT: {res.terminal_args.get('summary', '')}"
        report += ("\nFailures: " + "; ".join(failures)) if failures else "\nFailures: none"
        return report
    return f"EXECUTOR DID NOT REPORT ({res.last_text}). The plan may be only partially applied."


async def run_reviewer(session, all_tools, usage, *, task: str, last_plan: Optional[list],
                       previous_review: Optional[dict], focus: str) -> dict:
    tools = select_tools(all_tools, REVIEWER_TOOLKIT) + [SUBMIT_REVIEW_TOOL]
    context = [f"Original task given to the team:\n{task}"]
    if last_plan:
        context.append("Last plan sent to the executor:\n" + json.dumps(last_plan, indent=2))
    if previous_review:
        context.append("Your previous review (check whether the issues were addressed):\n"
                       + format_review(previous_review))
    context.append(f"Supervisor's focus for this review: {focus or 'general assessment'}")

    messages = [
        {"role": "system", "content": REVIEWER_PROMPT},
        {"role": "user", "content": "\n\n".join(context)},
        image_message("Symbolic Reference Layout from datasheet.", REFERENCE_IMAGE),
    ]
    res = await run_agent_loop(
        role="reviewer", messages=messages, tools=tools, session=session, usage=usage,
        terminal_tools={"submit_review"}, max_iter=REVIEWER_MAX_ITER, show_board=True,
    )
    if res.terminal_tool == "submit_review":
        return res.terminal_args
    return {"verdict": "revise", "score": 0, "issues": [],
            "summary": f"Reviewer failed to produce a review: {res.last_text}"}


# ─────────────────────────────────────────────
# Supervisor / orchestration
# ─────────────────────────────────────────────

@dataclass
class RunState:
    rounds: int = 0
    reviews: int = 0
    plans: list = field(default_factory=list)
    last_review: Optional[dict] = None


DEFAULT_TASK = (
    "Execute a placement for the dcdc converter project to which you are connected via MCP. "
    "The placement for J1, J2 and U1 is fixed, so do not move them! Arrange the other components "
    "the way a power electronics PCB design expert would do. "
    "Consider the provided reference layout! "
    "Do not change the board outline! Keep all components on the top side! "
    "The y-axis is increasing downwards, keep this in mind! "
    "Make sure that in the final result all components are within the board boundary, but do not overlap! "
    "Use the board image for spatial reasoning instead of verifying every coordinate numerically."
)


async def supervisor_run(session: ClientSession, all_tools: list[dict], task: str = DEFAULT_TASK) -> dict:
    usage = Usage()
    state = RunState()

    # ---- handlers that spawn sub-agents ----
    async def h_execute(args: dict) -> str:
        if state.rounds >= MAX_ROUNDS:
            return (f"BUDGET EXHAUSTED: {MAX_ROUNDS} executor rounds already used. "
                    "You cannot place anything else. Call `finish`.")
        placements = args.get("placements") or []
        if not placements:
            return "Error: `placements` is empty."
        bad = [p.get("reference") for p in placements
               if str(p.get("reference", "")).upper() in FIXED_COMPONENTS]
        if bad:
            return (f"Error: {bad} are fixed components and must not be moved. "
                    "Resubmit the plan without them (no round was consumed).")

        state.rounds += 1
        state.plans.append(placements)
        print(f"[supervisor] executor round {state.rounds}/{MAX_ROUNDS}")
        report = await run_executor(session, all_tools, usage, placements, args.get("rationale", ""))
        drc = await count_drc_errors(session)
        return (f"{report}\nDRC errors after this step: {drc}\n"
                f"(executor rounds used: {state.rounds}/{MAX_ROUNDS}; see the updated board image)")

    async def h_review(args: dict) -> str:
        if state.reviews >= MAX_REVIEWS:
            return (f"BUDGET EXHAUSTED: {MAX_REVIEWS} reviews already used. "
                    "Decide yourself and call `finish` or `execute_placement`.")
        state.reviews += 1
        print(f"[supervisor] review {state.reviews}/{MAX_REVIEWS}")
        review = await run_reviewer(
            session, all_tools, usage, task=task,
            last_plan=state.plans[-1] if state.plans else None,
            previous_review=state.last_review, focus=args.get("focus", ""),
        )
        state.last_review = review
        return format_review(review) + f"\n(reviews used: {state.reviews}/{MAX_REVIEWS})"

    # ---- supervisor setup ----
    tools = (select_tools(all_tools, SUPERVISOR_TOOLKIT)
             + [EXECUTE_PLACEMENT_TOOL, REQUEST_REVIEW_TOOL, SUPERVISOR_FINISH_TOOL])

    messages = [
        {"role": "system", "content": SUPERVISOR_PROMPT},
        {"role": "user", "content": f"User Query: {task}"},
        image_message("Symbolic Reference Layout from datasheet.", REFERENCE_IMAGE),
    ]

    res = await run_agent_loop(
        role="supervisor", messages=messages, tools=tools, session=session, usage=usage,
        terminal_tools={"finish"},
        handlers={"execute_placement": h_execute, "request_review": h_review},
        max_iter=SUPERVISOR_MAX_ITER, show_board=True,
    )

    success = res.terminal_tool == "finish"
    summary = res.terminal_args.get("summary", "") if success else (
        res.last_text or "Supervisor ended without calling finish.")

    # Safety net: make sure the final state is persisted
    try:
        await session.call_tool("save_project", {})
    except Exception as e:
        print(f"[orchestrator] final save_project failed: {e}")

    violations = await count_drc_errors(session)

    return {
        "success": success,
        "summary": summary,
        "tokens": usage.total,
        "tokens_by_role": usage.by_role,
        "violations": violations,
        "rounds": state.rounds,
        "reviews": state.reviews,
    }


async def agent_session(project_path: str) -> dict:
    server_params = StdioServerParameters(
        command="podman",
        args=[
            "run", "--rm", "-i",
            "-v", f"{os.environ['HOME']}/kicad_workspace:/workspace/projects:Z",
            "kicad-mcp",
            "node", "/workspace/KiCAD-MCP-Server/dist/index.js",
        ],
        env={},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            all_tools = [mcp_tool_to_openai(t) for t in mcp_tools.tools]
            print(f"[orchestrator] {len(all_tools)} tools available")

            print("[orchestrator] opening the project...")
            await session.call_tool("open_project", {"filename": project_path})
            print("[orchestrator] project is loaded.")

            return await supervisor_run(session, all_tools)


if __name__ == "__main__":

    path = <PATH> # path to .kicad_pro file
    
    result = asyncio.run(agent_session(project_path=path))
        