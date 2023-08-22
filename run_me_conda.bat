@echo off

:: Activate the Anaconda environment
call activate TEManalyzer

:: Run the Python script
python TEMGUI.py

:: Check if the "deactivate" parameter is provided
if "%~1"=="deactivate" (
    call conda deactivate
)

:: Pause the batch script so that you can see the output before it closes (optional)
pause