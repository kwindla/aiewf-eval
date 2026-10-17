"""Run an analyzer with writes confined to this worktree and networking disabled."""

import sys, os, runpy, json
from pathlib import Path

root = Path.cwd().resolve()
reads = set()


def audit(event, args):
    if event == "open":
        path, mode, flags = args
        if isinstance(path, (str, bytes)):
            p = Path(os.fsdecode(path)).resolve()
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                if not p.is_relative_to(root):
                    raise RuntimeError("Write outside worktree: " + str(p))
            elif p.is_file() and "/runs/" in str(p):
                reads.add(str(p))
    if event in ("os.rename", "os.remove", "os.mkdir", "os.rmdir"):
        for path in args[:2] if event == "os.rename" else args[:1]:
            if isinstance(path, (str, bytes)) and not Path(
                os.fsdecode(path)
            ).resolve().is_relative_to(root):
                raise RuntimeError("Mutation outside worktree: " + str(path))
    if event in ("socket.connect", "subprocess.Popen"):
        raise RuntimeError("Offline analysis forbids " + event)


sys.addaudithook(audit)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
sys.path.insert(0, str(root))
sys.dont_write_bytecode = True
runpy.run_path(script, run_name="__main__")
