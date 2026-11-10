from pathlib import Path

from xhtml2pdf import pisa


root = Path(__file__).resolve().parent
source = root / "eth_15m_strategy_report.html"
output = root / "eth_15m_strategy_report.pdf"

with output.open("wb") as pdf_file:
    result = pisa.CreatePDF(source.read_bytes(), dest=pdf_file, encoding="utf-8")

if result.err:
    raise SystemExit(f"PDF conversion failed with {result.err} error(s)")

print(f"Created {output} ({output.stat().st_size} bytes)")