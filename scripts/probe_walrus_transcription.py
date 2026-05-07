"""Probe gpt-transcribe-alpha-walrus-2 as the input_audio_transcription model.

Connects to the Realtime API with gpt-realtime-alpha-dolphin-14, then sends
a session.update setting audio.input.transcription.model to walrus-2.
Reports whether the server accepts the config and echoes it back.
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv

import websockets


load_dotenv()

REALTIME_MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-realtime-alpha-dolphin-14"
ASR_MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt-transcribe-alpha-walrus-2"
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


async def main():
    print(f"=== probing {REALTIME_MODEL} with input_audio_transcription={ASR_MODEL} ===")
    async with await open_ws() as ws:
        evt = await recv_until(ws, lambda e: e.get("type") == "session.created", timeout=15)
        if not evt:
            print("ERROR: no session.created received")
            return
        print("default audio.input.transcription:")
        print(json.dumps(evt["session"].get("audio", {}).get("input", {}).get("transcription"), indent=2))
        print()

        # Try the nested-shape session.update with a transcription model
        update = {
            "type": "realtime",
            "audio": {
                "input": {
                    "transcription": {"model": ASR_MODEL},
                }
            },
        }
        print(f"sending session.update with: {json.dumps(update)}")
        await ws.send(json.dumps({"type": "session.update", "session": update}))
        result = await recv_until(
            ws,
            lambda e: e.get("type") in ("session.updated", "error"),
            timeout=15,
        )
        if result is None:
            print("ERROR: timed out waiting for response")
            return
        if result["type"] == "error":
            err = result.get("error", {})
            print(f"REJECTED: {err.get('code')}: {err.get('message')}")
            return
        echoed = (
            result.get("session", {})
            .get("audio", {})
            .get("input", {})
            .get("transcription")
        )
        print(f"ACCEPTED. server echoed: {json.dumps(echoed)}")


if __name__ == "__main__":
    asyncio.run(main())
