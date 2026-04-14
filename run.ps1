$pythonPath = "C:\Users\Ankit\AppData\Local\Programs\Python\Python39\python.exe"
if (Test-Path $pythonPath) {
    Write-Host "Launching SRR-DFS using Python 3.9..." -ForegroundColor Cyan
    & $pythonPath -m streamlit run streamlit_app.py
} else {
    Write-Host "Error: Python 3.9 not found. Attempting default python..." -ForegroundColor Yellow
    python -m streamlit run streamlit_app.py
}
