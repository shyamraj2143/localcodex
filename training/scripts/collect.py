import argparse, json
from training.pipeline import RAW, append_record, trajectory

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",help="JSON report file to import"); a=p.parse_args()
    if a.input:
        with open(a.input,encoding="utf-8") as f: record=json.load(f)
        append_record(RAW,trajectory(record)); print(RAW)
    else: print("Automatic collection writes completed agent task reports to the workspace .localcodex/memory/trajectories.jsonl; import with --input.")
if __name__=="__main__": main()
