# Gemma 4 31B on Cerebras — quickstart for benchmarks on EC2

This note gets you running Gemma 4 31B against Cerebras Inference from a fresh
EC2 instance in us-east-1 (same region as Cerebras, so RTT is minimal). Drop it
into a new Claude session in another repo to bootstrap.

## What the model is

| Field | Value |
|---|---|
| Cerebras model ID | `gemma-4-31b` |
| API | OpenAI-compatible Chat Completions at `https://api.cerebras.ai/v1` |
| Context | 131K tokens |
| Spec gen speed | ~1,700 TPS |
| Reasoning default | **OFF** (set `reasoning_effort=low/medium/high` to enable; `none` keeps it off) |
| Note on levels | low/medium/high are equivalent for Gemma 4 today |
| Parallel tool calls | supported (`parallel_tool_calls=True`) |
| Structured output | `response_format` with `json_schema`, `strict: true` |
| Streaming | `stream=True` (tool calls stream too, but no `tool_stream`) |
| Text-only at launch | image input planned, no video |

Cerebras's recommended sampling (from their guide):

| Use case | Temperature | Top P | Reasoning |
|---|---:|---:|---|
| Agentic workflows | 0.8 | 0.95 | `medium` |
| Fast Q&A / summarization | 0.6 | 0.95 | `none` |
| Math/coding | 0.8 | 0.95 | `medium` or `high` |
| Deterministic / structured | 0.3 | 0.95 | `none` |

## Credentials

You'll need a Cerebras EA key with `gemma-4-31b` access. Confirm with:

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key="<csk-...>")
print(sorted(m.id for m in client.models.list().data))
# expect "gemma-4-31b" in the list
```

If you only see `gpt-oss-120b`, `zai-glm-4.7`, `llama3.1-8b` etc., that key
doesn't have Gemma — ask Cerebras to entitle it.

## Minimal direct-call example

```python
from openai import OpenAI
c = OpenAI(base_url="https://api.cerebras.ai/v1", api_key="<csk-...>")

# Reasoning OFF (fast/cheap path)
r = c.chat.completions.create(
    model="gemma-4-31b",
    messages=[{"role":"user","content":"…"}],
    temperature=0.6, top_p=0.95,
    reasoning_effort="none",
    max_completion_tokens=1024,
)

# Reasoning ON (agentic / multi-step)
r = c.chat.completions.create(
    model="gemma-4-31b",
    messages=[{"role":"user","content":"…"}],
    temperature=0.8, top_p=0.95,
    reasoning_effort="medium",
    max_completion_tokens=4096,
)
print(r.choices[0].message.reasoning)  # reasoning trace (Cerebras renamed from reasoning_content)
print(r.choices[0].message.content)
```

Streaming TTFB: with thinking on, the first chunk you'll see is a
`delta.reasoning` token, not user-visible content. If you want **content-first
TTFT**, watch for `chunk.choices[0].delta.content` and ignore reasoning-only
deltas. (This matters for voice-agent latency budgets.)

## Spinning up EC2 in us-east-1 (Daily AWS)

The benefit of running from us-east-1 is small but real: ~50–100 ms saved per
warm round-trip (Cerebras serves from us-east-1). For prompts with large prefill
(e.g. 10K+ token KBs), the network savings are dominated by server-side prefill
and you may only see ~50 ms median TTFT improvement; for short prompts you can
save ~400 ms. Measure first if it matters.

### Local prereqs

- `aws` CLI v2 installed, Daily SSO configured
- An SSH key with **no passphrase** (interactive prompts don't work in tool sessions)

If your `~/.ssh/id_ed25519` has a passphrase, generate a throwaway:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/aws-bench-key -C "aws-bench"
```

### SSO + profile

Daily Demos = account `419599258555`, profile
`AWSAdministratorAccess-419599258555` (sso_session `khk`). Other Daily profiles
are listed in `~/.aws/config` if you need a different account.

```bash
aws sso login --sso-session khk  # interactive (browser)
export AWS_PROFILE=AWSAdministratorAccess-419599258555
export AWS_REGION=us-east-1
aws sts get-caller-identity  # should show your account
```

### Launch a c7i.large

```bash
# Latest AL2023 x86_64 AMI
AMI=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

# SSH ingress from your current IP only
MY_IP=$(curl -s https://checkip.amazonaws.com)/32
SG_ID=$(aws ec2 create-security-group --group-name <name>-sg \
  --description "SSH for bench" --vpc-id $(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text) \
  --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 22 --cidr $MY_IP

# Import key + launch
aws ec2 import-key-pair --key-name <name> \
  --public-key-material "fileb://$HOME/.ssh/aws-bench-key.pub"

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI --instance-type c7i.large \
  --key-name <name> --security-group-ids $SG_ID \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=16,VolumeType=gp3,DeleteOnTermination=true}' \
  --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=<name>},{Key=Owner,Value=you@daily.co}]' \
  --query 'Instances[0].InstanceId' --output text)

aws ec2 wait instance-running --instance-ids $INSTANCE_ID
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# Wait for SSH (instance-running != ssh-ready; takes ~30-60s more)
for i in $(seq 1 24); do
  ssh -i ~/.ssh/aws-bench-key -o StrictHostKeyChecking=no -o BatchMode=yes \
      -o ConnectTimeout=5 ec2-user@$PUBLIC_IP true 2>/dev/null && break
  sleep 5
done
```

Don't forget to tear down when done:

```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 delete-security-group --group-id $SG_ID  # after instance fully terminates
aws ec2 delete-key-pair --key-name <name>
```

### Bootstrap the instance

```bash
ssh -i ~/.ssh/aws-bench-key ec2-user@$PUBLIC_IP <<'REMOTE'
# uv (manages Python and deps cleanly)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Pin Python to 3.12 (3.14 is default in current uv but numba breaks on it)
~/.local/bin/uv python install 3.12
mkdir -p ~/work && cd ~/work
# Project bootstrap goes here (your benchmark repo). Example for the aiewf-eval style:
#   git clone <repo>
#   cd <repo>
#   ~/.local/bin/uv python pin 3.12
#   ~/.local/bin/uv sync
REMOTE

# Copy your .env (with CEREBRAS_API_KEY and any other secrets)
scp -i ~/.ssh/aws-bench-key /path/to/.env ec2-user@$PUBLIC_IP:~/work/<repo>/.env
```

## Caveats from prior work

- **EC2 doesn't ship with a Claude CLI**, so if your benchmark judge uses
  `claude-agent-sdk`, the SDK falls back to its bundled `claude` binary. That
  binary needs `ANTHROPIC_API_KEY` exported (it doesn't auto-read `.env`):
  ```bash
  ANTHROPIC_API_KEY=$(grep '^ANTHROPIC_API_KEY=' .env | cut -d= -f2) \
    uv run <judge command>
  ```
  Same trick works locally if your laptop's Claude CLI is stuck in a nested
  session and `claude-agent-sdk` calls fail with "Fatal error in message
  reader" / "Command failed with exit code 1".

- **`HttpTokens=required`** (IMDSv2) is fine for c7i and cloud-init's SSH key
  install. Leave it on.

- **EAP endpoint concurrency** — on at least one Cerebras EA org, ≥10–40
  concurrent requests caused hangs/idle-timeouts around turn 8–9 of a multi-turn
  benchmark (no clean 429s, just stalls). Cap concurrency at **≤8** for any large
  sweep on EAP keys.

- **Sanity-check first** with `--only-turns 0,1,2` (or your benchmark's
  equivalent) before launching a big batch. Cold-start TTFB on first request is
  often 5–10× the warm number; the second and third calls are representative.

## Cerebras-side reasoning notes

- Default is reasoning OFF. Sending no `reasoning_effort` field works (uses the
  model's default = off). Sending `reasoning_effort="none"` is the explicit form.
- low/medium/high all enable reasoning; today they're equivalent on Gemma 4
  according to the EA guide.
- Reasoning traces stream first (as `delta.reasoning`), then content (as
  `delta.content`). For voice-agent latency math, **don't stop your TTFT timer
  on the first chunk** — wait for the first non-empty `delta.content`.
