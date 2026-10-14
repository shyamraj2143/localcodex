import argparse
from training.pipeline import read_jsonl, write_jsonl, safe, SECRET_FILE

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--exclude",action="append",default=[]); a=p.parse_args()
    rows=read_jsonl(a.input); exclusions=[x.casefold() for x in a.exclude]
    out=[]
    for row in rows:
        if any(x in str(row).casefold() for x in exclusions): continue
        row=safe(row)
        for key in list(row):
            if SECRET_FILE.search(str(key)): row.pop(key,None)
        out.append(row)
    write_jsonl(a.output,out); print(f"Sanitized {len(out)} records to {a.output}; review output for undetected secrets.")
if __name__=="__main__": main()
