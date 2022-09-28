start cmd.exe /k "python betavision-screenshot.py"
start cmd.exe /k "python betavision-detect.py"
timeout /t -1
start cmd.exe /k "python betavision-censor.py"