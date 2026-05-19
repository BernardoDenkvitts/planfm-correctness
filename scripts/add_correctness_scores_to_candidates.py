"""Add correctness score in each ./data/correctness_dataset/candidates/*.jsonl"""

import json
from pathlib import Path
from code.downstream.plan_utils import compute_correctness_score, load_problem_context

def main():
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    candidates_dir = data_dir / "correctness_dataset" / "candidates"
    
    for jsonl_path in candidates_dir.glob("*.jsonl"):
        print(f"Processing {jsonl_path.name}...")
        
        # Read the candidates
        candidates = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                candidates.append(json.loads(line))
                
        # Update with correctness_score
        updated = []
        for c in candidates:
            domain = c["domain"]
            split = c["split"]
            problem = c["problem"]
            plan = c["plan"]
            
            # Load the context
            context = load_problem_context(
                data_dir=data_dir,
                domain=domain,
                split=split,
                problem=problem
            )
            
            # Compute score
            score = compute_correctness_score(context, plan)
            c["correctness_score"] = score
            updated.append(c)
            
        # Write them back
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for c in updated:
                f.write(json.dumps(c) + "\n")
                
    print("Done! You can now run build_correctness_dataset with --skip_candidates.")

if __name__ == "__main__":
    main()
