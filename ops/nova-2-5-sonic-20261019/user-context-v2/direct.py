"""Verify the preserved successful native model/tool probe; do not repeat it."""
import json
from common import HERE,MODEL,sha
async def main():
    path=HERE.parent/'smoke/direct.json'
    result=json.loads(path.read_text())
    assert result['verified'] and result['tool_result_sent'] and result['invoked_model']==MODEL
    assert result['calls'][0]['toolName']=='probe'
    print('Preserved native model/tool probe verified: '+sha(path),flush=True)
