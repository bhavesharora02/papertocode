# scripts/auto_verify.py
import argparse, os, sys, json, re
from core.auto_repair.engine import AutoRepairEngine
from core.auto_repair.utils import run_cmd
from core.auto_repair.rules import remove_unicode_from_verify_repo
import google.generativeai as genai

# -----------------------------------------------------
# ? Gemini setup
# -----------------------------------------------------
if not os.getenv("GEMINI_API_KEY"):
    print("Warning:  Warning: GEMINI_API_KEY not found. Gemini fallback will be disabled.")
else:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# -----------------------------------------------------
# ? Gemini reasoning for unseen tracebacks
# -----------------------------------------------------
def call_gemini_for_patch(error_text, file_hint="verify_repo.py"):
    """
    Ask Gemini 2.5 Pro to analyze unseen tracebacks
    and suggest a minimal, safe patch in strict JSON format.
    """
    prompt = f"""
You are an expert AI debugging and code repair system.
Below is a real Python traceback from a machine-learning research automation pipeline.
Your goal: suggest a minimal, safe fix that would make the code run successfully
without changing its core logic.

---
ERROR TRACEBACK:
{error_text}
---

Respond strictly in JSON, like this:
{{
  "file": "<relative path to file to patch>",
  "before": "<substring to find>",
  "after": "<replacement text>"
}}
If no clear fix is possible, return: {{ "file": null }}
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        match = re.search(r'\{.*\}', text, re.S)
        if match:
            patch = json.loads(match.group(0))
            print(f"? Gemini suggestion:\n{json.dumps(patch, indent=2)}")
            return patch
        print("Warning:  Gemini returned unstructured output.")
    except Exception as e:
        print(f"Warning:  Gemini API call failed: {e}")
    return None


# -----------------------------------------------------
# ? Apply Gemini patch
# -----------------------------------------------------
def apply_patch(patch):
    """Apply Gemini?s suggested patch to the specified file."""
    if not patch or not patch.get("file"):
        print("Warning:  Gemini returned no actionable patch.")
        return False

    file_path = patch["file"]
    if not os.path.exists(file_path):
        alt = os.path.join("scripts", file_path)
        if os.path.exists(alt):
            file_path = alt
        else:
            print(f"Warning:  File {file_path} not found.")
            return False

    with open(file_path, "r", encoding="utf-8") as f:
        src = f.read()

    if patch["before"] in src:
        new_src = src.replace(patch["before"], patch["after"])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_src)
        print(f"OK: Gemini patch applied to {file_path}")
        return True
    else:
        print(f"Warning:  Could not find target text in {file_path}")
        return False


# -----------------------------------------------------
# ? Main Auto Verify Logic
# -----------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ir_path")
    p.add_argument("repo_dir")
    p.add_argument("report_path")
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    engine = AutoRepairEngine(debug=args.debug)

    cmd = [
        sys.executable, "-m", "scripts.verify_repo",
        args.ir_path, args.repo_dir, args.report_path
    ]
    if args.debug:
        cmd.append("--debug")

    # 1?? First: direct verification
    rc, out, err = run_cmd(cmd)

    # OK: If process succeeded but no visible output, treat as success
    if rc == 0 and not (out.strip() or err.strip()):
        print("OK: Verification completed successfully (no explicit output).")
        print("OK: Auto-verify: success on first run")
        if os.path.exists(args.report_path):
            with open(args.report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            print("\n? Final Verification Report:")
            print(json.dumps(report, indent=2))
            return

    # Warning: If failed but no logs, force debug mode
    if rc != 0 and not (err or out):
        print("Warning:  No stdout/stderr captured ? retrying with forced debug...")
        cmd.append("--debug")
        rc, out, err = run_cmd(cmd)

    last_error = (err + "\n" + out).strip() if (err or out) else "Empty traceback (no output)"

    # 2?? Rule-based and Gemini-assisted repair loop
    for attempt in range(1, args.max_retries + 1):
        print(f"\n??  Auto-repair attempt {attempt}/{args.max_retries}")
        print("?? error snippet ???????????????????????????")
        print("\n".join(last_error.splitlines()[-25:]))
        print("?????????????????????????????????????????????")

        repaired = engine.repair(
            error_text=last_error,
            ir_path=args.ir_path,
            repo_dir=args.repo_dir,
        )

        if not repaired:
            print("Warning:  No rule matched or repair failed. Trying Gemini fallback...")
            patch = call_gemini_for_patch(last_error)
            if patch and apply_patch(patch):
                # ? Clean Unicode before regenerating
                remove_unicode_from_verify_repo(args.ir_path, args.repo_dir)

                print("? Regenerating repo after Gemini fix...\n")
                gen_cmd = [sys.executable, "-m", "scripts.generate_repo", args.ir_path]
                grc, gout, gerr = run_cmd(gen_cmd)
                if grc != 0:
                    print("Error: Repo regeneration failed after Gemini patch:")
                    print(gout or gerr)
                    sys.exit(1)

                # OK: Always verify after regeneration
                print("? Running final verification after Gemini regeneration...\n")
                rc, out, err = run_cmd(cmd)
                if rc == 0:
                    print(out.strip())
                    print("OK: Auto-verify: success after Gemini fix (final verification).")
                    return
                else:
                    print("Error: Verification still failing after regeneration.")
                    print(err or out)
                    sys.exit(1)
            else:
                print("Error: Gemini could not provide a fix. Exiting.")
                sys.exit(1)

        # ? Regenerate repo if IR changed
        if repaired.get("regenerate_repo", False):
            remove_unicode_from_verify_repo(args.ir_path, args.repo_dir)
            gen_cmd = [sys.executable, "-m", "scripts.generate_repo", args.ir_path]
            grc, gout, gerr = run_cmd(gen_cmd)
            if grc != 0:
                print("Error: Repo regeneration failed.")
                print(gout, gerr)
                sys.exit(1)

        # ? Retry verification
        rc, out, err = run_cmd(cmd)
        if rc == 0:
            print(out.strip())
            print("OK: Auto-verify: success after repair")
            return

        last_error = (err or out) or "Unknown failure"

    # 3?? Fallback ? final Gemini attempt
    print("Error: Auto-verify failed after all rule-based repair attempts.")
    print("??  Invoking Gemini fallback...\n")

    patch = call_gemini_for_patch(last_error)
    if patch and apply_patch(patch):
        remove_unicode_from_verify_repo(args.ir_path, args.repo_dir)
        print("? Regenerating repo after Gemini fix...\n")
        gen_cmd = [sys.executable, "-m", "scripts.generate_repo", args.ir_path]
        grc, gout, gerr = run_cmd(gen_cmd)
        if grc != 0:
            print("Error: Repo regeneration failed after Gemini patch:")
            print(gout or gerr)
            sys.exit(1)

        # OK: Always verify one last time
        print("? Running final verification after Gemini regeneration...\n")
        rc, out, err = run_cmd(cmd)
        if rc == 0:
            print(out.strip())
            print("OK: Verification succeeded after Gemini repair (final pass).")
            return
        else:
            print("Error: Even Gemini patch failed. Check logs.")
            print(err or out)
            sys.exit(1)
    else:
        print("Error: Gemini returned no actionable patch. Manual review required.")
        sys.exit(1)


if __name__ == "__main__":
    main()
