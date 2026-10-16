#!/usr/bin/env bash
# Render the executed notebook:
#   field_boundary_comparison.html          — full report with code and outputs
#   field_boundary_comparison_report.pdf    — report without code (text, tables, figures)
set -euo pipefail
cd "$(dirname "$0")"
NB=${1:-field_boundary_comparison}
JUP=.venv/bin/jupyter

$JUP nbconvert --to html "$NB.ipynb" --output "$NB"

# Code-free HTML, then print to PDF with headless Chrome
$JUP nbconvert --to html --no-input --no-prompt "$NB.ipynb" --output "${NB}_report"
python3 - "${NB}_report.html" <<'EOF'
import sys
p = sys.argv[1]
css = """<style>
@page { size: A4; margin: 14mm 12mm; }
body { font-size: 10.5pt; }
img { max-width: 100% !important; height: auto !important; page-break-inside: avoid; }
table { font-size: 8.5pt; page-break-inside: avoid; }
h1, h2, h3 { page-break-after: avoid; }
.jp-OutputArea-output pre { white-space: pre-wrap; font-size: 8pt; }
</style>"""
s = open(p, encoding="utf-8").read().replace("</head>", css + "</head>", 1)
open(p, "w", encoding="utf-8").write(s)
EOF
google-chrome --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer \
  --virtual-time-budget=20000 --print-to-pdf="${NB}_report.pdf" "file://$PWD/${NB}_report.html" 2>/dev/null
rm -f "${NB}_report.html"
ls -la "$NB.html" "${NB}_report.pdf"
