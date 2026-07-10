@echo off
cd /d "%~dp0"
py -3.12 -m streamlit run app.py --server.port 8501
