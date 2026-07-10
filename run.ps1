# CrossMind launcher — always uses Python 3.12
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

py -3.12 -m streamlit run app.py --server.port 8501
