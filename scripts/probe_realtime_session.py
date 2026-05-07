"""Raw WebSocket probe for OpenAI Realtime API session config.

Connects to a Realtime model, prints the session.created payload, then
sweeps reasoning effort values via session.update and reports which the
server accepts.
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv

import websockets


load_dotenv()

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-realtime-alpha-dolphin-14"
EFFORT_LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh"]
URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

API_KEY = os.environ["OPENAI_API_KEY"]


async def recv_until(ws, predicate, timeout=10.0):
    """Receive frames until predicate(evt) is True or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            return None
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        evt = json.loads(raw)
        if predicate(evt):
            return evt


async def open_ws():
    return await websockets.connect(
        URL,
        additional_headers={
            "Authorization": f"Bearer {API_KEY}",
        },
        max_size=10 * 1024 * 1024,
    )


async def probe_default():
    print(f"=== {MODEL}: session.created defaults ===")
    async with await open_ws() as ws:
        evt = await recv_until(ws, lambda e: e.get("type") == "session.created", timeout=15)
        if not evt:
            print("ERROR: no session.created received")
            return None
        session = evt.get("session", {})
        print(json.dumps(session, indent=2))
        return session


async def probe_effort(payload_shape, effort, label):
    """Try a session.update with the given payload shape and effort value."""
    if payload_shape == "flat":
        update = {"type": "realtime", "reasoning_effort": effort}
    elif payload_shape == "nested":
        update = {"type": "realtime", "reasoning": {"effort": effort}}
    else:
        raise ValueError(payload_shape)

    async with await open_ws() as ws:
        await recv_until(ws, lambda e: e.get("type") == "session.created", timeout=15)
        await ws.send(json.dumps({
            "type": "session.update",
            "session": update,
        }))
        evt = await recv_until(
            ws,
            lambda e: e.get("type") in ("session.updated", "error"),
            timeout=15,
        )
    if evt is None:
        return label, effort, "TIMEOUT", None
    if evt["type"] == "error":
        err = evt.get("error", {})
        return label, effort, "ERROR", f"{err.get('code')}: {err.get('message')}"
    sess = evt.get("session", {})
    echoed_flat = sess.get("reasoning_effort")
    echoed_nested = (sess.get("reasoning") or {}).get("effort")
    return label, effort, "OK", f"flat={echoed_flat!r} nested={echoed_nested!r}"


async def main():
    default_sess = await probe_default()

    print()
    print(f"=== reasoning effort sweep for {MODEL} ===")
    results = []
    for shape in ("flat", "nested"):
        label = "reasoning_effort" if shape == "flat" else "reasoning.effort"
        for effort in EFFORT_LEVELS:
            try:
                r = await probe_effort(shape, effort, label)
            except Exception as e:
                r = (label, effort, "EXC", str(e))
            results.append(r)
            print(f"  {r[0]:20s} {r[1]:8s} -> {r[2]:7s} {r[3] or ''}")

    print()
    print("=== summary ===")
    if default_sess is not None:
        print(f"default reasoning_effort = {default_sess.get('reasoning_effort')!r}")
        print(f"default reasoning        = {default_sess.get('reasoning')!r}")
    accepted = [(l, e) for (l, e, s, _) in results if s == "OK"]
    rejected = [(l, e, d) for (l, e, s, d) in results if s != "OK"]
    print(f"accepted ({len(accepted)}):")
    for l, e in accepted:
        print(f"  {l} = {e}")
    print(f"rejected ({len(rejected)}):")
    for l, e, d in rejected:
        print(f"  {l} = {e}: {d}")


if __name__ == "__main__":
    asyncio.run(main())
