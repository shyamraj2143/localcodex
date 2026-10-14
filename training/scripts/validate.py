import argparse, json
from training.pipeline import RAW, validate

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default=str(RAW)); a=p.parse_args(); report=validate(a.input); print(json.dumps(report,indent=2)); raise SystemExit(0 if report["valid"] else 2)
if __name__=="__main__": main()
