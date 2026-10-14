import argparse, hashlib, json
from pathlib import Path
from training.pipeline import REVIEWED, PROCESSED, read_jsonl, split, write_jsonl

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default=str(REVIEWED/"approved.jsonl")); p.add_argument("--output-dir",default=str(PROCESSED)); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    rows=read_jsonl(a.input); parts=split([r for r in rows if r.get("review_status")=="approved" and r.get("final_status")=="success"],a.seed); dest=Path(a.output_dir); dest.mkdir(parents=True,exist_ok=True)
    for name,items in parts.items(): write_jsonl(dest/f"{name}.jsonl",items)
    digest=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(); manifest={"source":str(Path(a.input).resolve()),"source_sha256":digest,"seed":a.seed,"grouping":"duplicate_group or request+changed_files SHA-256","counts":{k:len(v) for k,v in parts.items()},"format_version":1}; (dest/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
