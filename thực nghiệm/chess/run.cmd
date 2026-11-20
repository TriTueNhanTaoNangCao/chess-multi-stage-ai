@echo off
setlocal
pushd "%~dp0..\.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Create the project Python environment first.
  popd
  exit /b 1
)
".venv\Scripts\python.exe" "scripts\run_all_models.py" %*
set "benchmark_exit=%errorlevel%"
popd
exit /b %benchmark_exit%
