@echo off
powershell.exe -NoExit -Command "Set-Location -LiteralPath '%~dp0'; streamlit run App_st.py"