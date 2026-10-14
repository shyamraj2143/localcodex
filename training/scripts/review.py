import argparse
from training.pipeline import RAW, review

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default=str(RAW)); p.add_argument("--task-id"); p.add_argument("--decision",choices=("approved","rejected","pending")); p.add_argument("--note",default=""); a=p.parse_args()
    from training.pipeline import read_jsonl
    rows=read_jsonl(a.input)
    for row in rows:
        if not a.task_id or row.get("task_id")==a.task_id: print(row)
    if a.task_id and a.decision: review(a.input,a.task_id,a.decision,a.note)
if __name__=="__main__": main()
