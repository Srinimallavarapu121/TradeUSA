from pathlib import Path

from pypdf import PdfReader


pdf = Path(__file__).resolve().parent / "eth_15m_strategy_report.pdf"
reader = PdfReader(str(pdf))
text = "\n".join(page.extract_text() or "" for page in reader.pages)
print(
    f"pages={len(reader.pages)} bytes={pdf.stat().st_size} "
    f"title={'Strategy and Rules Report' in text} "
    f"gate={'DO NOT PROMOTE' in text}"
)