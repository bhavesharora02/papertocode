# scripts/codegen.py
import os, sys, json
from core.codegen.codegen import parse_pdf_to_ir  # OK: use your actual core function

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m scripts.codegen <pdf_path> <output_ir_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_ir = sys.argv[2]

    print(f"??  Step 1: Parsing paper -> {pdf_path}")
    try:
        ir_data = parse_pdf_to_ir(pdf_path)
    except Exception as e:
        print(f"Error:  Step 1 failed: could not parse paper: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_ir), exist_ok=True)
    with open(output_ir, "w", encoding="utf-8") as f:
        json.dump(ir_data, f, indent=2)

    print(f"OK:  IR successfully generated at {output_ir}")

if __name__ == "__main__":
    main()
