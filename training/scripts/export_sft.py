import argparse
from pathlib import Path
from training.pipeline import PROCESSED, export_sft, read_jsonl, write_jsonl

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default=str(PROCESSED/"train.jsonl")); p.add_argument("--output",default=str(PROCESSED/"train.sft.jsonl")); a=p.parse_args(); result=export_sft(read_jsonl(a.input)); write_jsonl(a.output,result); print(f"Exported {len(result)} verified SFT examples to {a.output}")
if __name__=="__main__": main()
