@echo off
REM Windows wrapper for the standalone revision bootstrap script.
REM Revision skeleton alignment:
REM - keeps the reproducible first-round environment bootstrap executable on Windows

powershell -ExecutionPolicy Bypass -File "%~dp0bootstrap_revision_env.ps1" %*
