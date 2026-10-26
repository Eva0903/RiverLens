"""Weather interpretation. Default overview never fits or recovers a model."""
from __future__ import annotations
import argparse
from jinjiang.config import load_config
from jinjiang.interpretation_random import InterpretationRunner,validate_spec

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/interpretation_random.json")
    parser.add_argument("--mode",choices=["audit","overview","explain"],default="overview")
    parser.add_argument("--replay-missing",action="store_true",help="Explicitly recover missing original checkpoints; this DOES train.")
    parser.add_argument("--max-replays",type=int,default=1)
    parser.add_argument("--max-cases",type=int)
    args=parser.parse_args()
    if args.max_replays<0 or (args.max_cases is not None and args.max_cases<0):
        parser.error("Task limits must be nonnegative.")
    if args.replay_missing and args.mode!="explain":
        parser.error("--replay-missing requires --mode explain.")
    c=load_config(args.config)
    try: validate_spec(c)
    except ValueError as exc: parser.error(str(exc))
    runner=InterpretationRunner(c)
    print(f"Finished: {runner.run(args.mode,args.replay_missing,args.max_replays,args.max_cases)}")

if __name__=="__main__": main()
