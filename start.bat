@echo off
echo Starting backend...
start "Backend" cmd /k "cd backend && venv\Scripts\activate && uvicorn app.main:app --reload --port 8000"

echo Starting frontend...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo Both servers starting in separate windows.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173

