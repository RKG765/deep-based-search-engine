import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
convert_report.py — Converts report.md → report.docx
with mermaid diagrams rendered as PNG images via mermaid.ink API.

Usage:
    python convert_report.py
"""

import re
import os
import base64
import httpx
import md2word

REPORT_MD   = "report.md"
REPORT_TEMP = "report_rendered.md"
REPORT_DOCX = "report.docx"
DIAGRAM_DIR = "diagrams"


def render_mermaid_to_png(mermaid_code: str, output_path: str) -> bool:
    """Render a Mermaid diagram to PNG using mermaid.ink API."""
    # Encode the mermaid code to base64 for the URL
    encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("utf-8")
    url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=white"

    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        if resp.status_code == 200 and len(resp.content) > 100:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            print(f"  [OK] Rendered: {output_path} ({len(resp.content)} bytes)")
            return True
        else:
            print(f"  [WARN] API returned status {resp.status_code} for diagram")
            return False
    except Exception as e:
        print(f"  [FAIL] Failed to render diagram: {e}")
        return False


def process_report():
    """Read report.md, render mermaid blocks as PNGs, write report_rendered.md."""
    os.makedirs(DIAGRAM_DIR, exist_ok=True)

    with open(REPORT_MD, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all ```mermaid ... ``` blocks
    pattern = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
    matches = list(pattern.finditer(content))

    print(f"[INFO] Found {len(matches)} mermaid diagrams")

    diagram_count = 0
    new_content = content

    # Process in reverse order to preserve character positions
    for match in reversed(matches):
        diagram_count += 1
        mermaid_code = match.group(1).strip()
        img_name = f"diagram_{diagram_count:02d}.png"
        img_path = os.path.join(DIAGRAM_DIR, img_name)

        success = render_mermaid_to_png(mermaid_code, img_path)

        if success:
            # Replace mermaid block with image reference
            abs_path = os.path.abspath(img_path).replace("\\", "/")
            replacement = f"![Diagram {diagram_count}]({abs_path})"
            new_content = new_content[:match.start()] + replacement + new_content[match.end():]
        else:
            # Keep as code block but label it
            replacement = f"**[Diagram {diagram_count}]** *(Mermaid diagram — see report.md for source)*\n\n```\n{mermaid_code}\n```"
            new_content = new_content[:match.start()] + replacement + new_content[match.end():]

    # Write the processed markdown
    with open(REPORT_TEMP, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"\n[INFO] Written {REPORT_TEMP} with {diagram_count} diagrams processed")

    # Convert to Word
    print(f"[INFO] Converting to {REPORT_DOCX}...")
    md2word.convert_file(REPORT_TEMP, REPORT_DOCX)
    print(f"\n[DONE] Output: {REPORT_DOCX}")

    # Cleanup temp file
    if os.path.exists(REPORT_TEMP):
        os.remove(REPORT_TEMP)
        print(f"[INFO] Cleaned up {REPORT_TEMP}")


if __name__ == "__main__":
    process_report()
