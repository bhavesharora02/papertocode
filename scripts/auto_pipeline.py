# scripts/auto_pipeline.py
import os, sys, json, subprocess, time

def run_step(cmd, desc):
    print(f"\n??  {desc} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error:  {desc} failed:\n{result.stderr or result.stdout}")
        sys.exit(1)
    else:
        print(f"OK:  {desc} completed.")
        return result.stdout.strip()

def auto_pipeline(pdf_path):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    ir_path = f"artifacts/ir/{base}_ir_patched.json"
    repo_dir = f"artifacts/repos/{base}_ir_patched"
    verify_report = f"artifacts/reports/{base}_verify.json"
    eval_report = f"artifacts/reports/{base}_eval.json"
    final_report = f"artifacts/reports/{base}_final.json"

    os.makedirs("artifacts/ir", exist_ok=True)
    os.makedirs("artifacts/repos", exist_ok=True)
    os.makedirs("artifacts/reports", exist_ok=True)

    # 1??  PDF -> IR
    run_step([
        sys.executable, "-m", "scripts.codegen",
        pdf_path, ir_path
    ], "Step 1: Paper parsed & IR generated")

    # 2??  IR -> Repo
    run_step([
        sys.executable, "-m", "scripts.generate_repo",
        ir_path, repo_dir
    ], "Step 2: Repository generated")

    # 3??  Auto Verify
    run_step([
        sys.executable, "-m", "scripts.auto_verify",
        ir_path, repo_dir, verify_report, "--debug"
    ], "Step 3: Auto verification (with Gemini fallback)")

    # 4??  Auto Evaluation
    run_step([
        sys.executable, "-m", "scripts.auto_eval",
        ir_path, repo_dir, eval_report
    ], "Step 4: Auto evaluation")

    # 5??  Combine reports
    try:
        with open(verify_report, "r") as f:
            verify_data = json.load(f)
        with open(eval_report, "r") as f:
            eval_data = json.load(f)
    except Exception as e:
        print(f"Warning:  Could not read reports: {e}")
        verify_data, eval_data = {}, {}

    final = {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paper": os.path.basename(pdf_path),
        "verify": verify_data,
        "eval": eval_data
    }

    with open(final_report, "w") as f:
        json.dump(final, f, indent=2)

    print("\n? FINAL REPORT:")
    print(json.dumps(final, indent=2))
    print(f"\nOK: Pipeline completed successfully! Final report -> {final_report}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.auto_pipeline <pdf_path>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    auto_pipeline(pdf_path)
