"""Run the standard harness with structural Responses lineage auditing."""
import importlib.util,sys
from pathlib import Path
from wire_audit import install
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('repair_standard_run_one',ROOT/'ops/aiewf-campaign-template/run_one.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
if __name__=='__main__':
 args=runner.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
 # The standard runner creates the directory itself; install a lazy audit writer.
 args.run_dir.rmdir();install(args.run_dir/'responses-history-audit.jsonl')
 raise SystemExit(runner.main())
