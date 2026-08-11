"""Probe gpt-transcribe-alpha-walrus-2 in turn_detection settings.

Peter @ OpenAI said walrus-2 is configured "in the turn detection settings".
Try a few shapes to see what the server accepts.
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv

import websockets


load_dotenv()

REALTIME_MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-realtime-alpha-dolphin-14"
ASR_MODEL = "gpt-transcribe-alpha-walrus-2"
URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
API_KEY = os.environ["OPENAI_API_KEY"]


async def recv_until(ws, predicate, timeout=10.0):
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
        additional_headers={"Authorization": f"Bearer {API_KEY}"},
        max_size=10 * 1024 * 1024,
    )


async def try_update(label, session_payload):
    print(f"--- {label} ---")
    print(f"  send: {json.dumps(session_payload)[:200]}")
    try:
        async with await open_ws() as ws:
            await recv_until(ws, lambda e: e.get("type") == "session.created", timeout=15)
            await ws.send(json.dumps({"type": "session.update", "session": session_payload}))
            evt = await recv_until(
                ws,
                lambda e: e.get("type") in ("session.updated", "error"),
                timeout=15,
            )
        if evt is None:
            print("  -> TIMEOUT")
            return
        if evt["type"] == "error":
            err = evt.get("error", {})
            print(f"  -> ERROR {err.get('code')}: {err.get('message')}")
        else:
            audio = evt.get("session", {}).get("audio", {})
            print(f"  -> OK; echoed audio.input.turn_detection={json.dumps(audio.get('input',{}).get('turn_detection'))}")
            print(f"     echoed audio.input.transcription={json.dumps(audio.get('input',{}).get('transcription'))}")
    except Exception as e:
        print(f"  -> EXC: {e}")


async def main():
    # Shape 1: server_vad with model field
    await try_update("server_vad + model=walrus-2", {
        "type": "realtime",
        "audio": {"input": {"turn_detection": {"type": "server_vad", "model": ASR_MODEL}}},
    })
    # Shape 2: semantic_vad type with model
    await try_update("semantic_vad + model=walrus-2", {
        "type": "realtime",
        "audio": {"input": {"turn_detection": {"type": "semantic_vad", "model": ASR_MODEL}}},
    })
    # Shape 3: turn_detection.type = walrus
    await try_update("turn_detection.type=walrus", {
        "type": "realtime",
        "audio": {"input": {"turn_detection": {"type": ASR_MODEL}}},
    })
    # Shape 4: pure semantic_vad (no model)
    await try_update("semantic_vad (no model)", {
        "type": "realtime",
        "audio": {"input": {"turn_detection": {"type": "semantic_vad"}}},
    })
    # Shape 5: server_vad with create_response (default)
    await try_update("server_vad with eagerness=auto", {
        "type": "realtime",
        "audio": {"input": {"turn_detection": {"type": "semantic_vad", "eagerness": "auto"}}},
    })


if __name__ == "__main__":
    asyncio.run(main())
