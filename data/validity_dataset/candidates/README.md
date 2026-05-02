# Candidate Plans

This folder contains one JSONL file per split.

Each row stores a candidate plan and its label:

- `candidate_id`: stable identifier built from domain, split, problem, variant index, and corruption type
- `domain`, `split`, `problem`: planning problem metadata
- `corruption_type`: `gold`, `truncate`, `delete`, `swap`, `replace`, `insert`, or `repeat`
- `plan`: grounded action sequence
- `plan_len` and `gold_plan_len`: candidate and reference plan lengths
- `label_valid`: binary plan-validity label
- `label_executable`: binary executability label

The full candidate set was labeled with VAL.
