import sys, json, os, subprocess
from core.debug.debugger import run_debugger, apply_patch
from core.ir.schema import IR

if __name__ == "__main__":
    ir_path = sys.argv[1]  # e.g. artifacts/ir/bert_ir.json
    repo_dir = sys.argv[2] # e.g. artifacts/repos/bert
    api_key = os.environ.get("GEMINI_API_KEY")

    for attempt in range(3):  # max 3 debugging iterations
        print(f"\n=== Debug Iteration {attempt+1} ===")

        # Load IR
        with open(ir_path) as f:
            ir_data = json.load(f)

        # Run verification
        subprocess.run(["python", "-m", "scripts.verify_repo", ir_path, repo_dir])

        report_path = ir_path.replace("artifacts/ir", "artifacts/reports").replace("_ir.json", "_verify.json")
        with open(report_path) as f:
            report = json.load(f)

        # Check pass/fail
        if report.get("pass") and all(report["pass"].values()):
            print("✅ Target metrics achieved! Exiting loop.")
            break
        else:
            print("❌ Target metrics not achieved, running debugger...")

        # Otherwise run debugger
        patch = run_debugger(ir_data, report, api_key)
        ir_data = apply_patch(ir_data, patch)

        # Save patched IR
        with open(ir_path, "w") as f:
            json.dump(ir_data, f, indent=2)

        # Regenerate repo
        subprocess.run(["python", "-m", "scripts.generate_repo", ir_path])
        print("Using GEMINI_API_KEY:", api_key[:6] + "..." if api_key else "NOT FOUND")

    print("🔁 Debug loop finished.")
