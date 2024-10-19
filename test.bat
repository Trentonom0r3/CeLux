@echo off
setlocal enabledelayedexpansion

REM Path to clang-format.exe
set CLANG_FORMAT_PATH="C:\Program Files\LLVM\bin\clang-format.exe"

REM Navigate to the project root directory
cd /d C:\Users\tjerf\source\repos\CeLux

REM Define directories to exclude (e.g., build directories)
set EXCLUDE_DIRS=out\build

REM Find and format all .cpp, .hpp, .cc, .h files excluding specified directories
for /r %%f in (*.cpp *.hpp *.cc *.h) do (
    set FILE_PATH=%%f
    set EXCLUDE=0
    for %%e in (%EXCLUDE_DIRS%) do (
        echo !FILE_PATH! | findstr /i /c:"\%%e\" >nul
        if !errorlevel! EQU 0 set EXCLUDE=1
    )
    if !EXCLUDE! EQU 0 (
        if exist "%%f" (
            echo Formatting %%f
            %CLANG_FORMAT_PATH% -i "%%f"
        ) else (
            echo File %%f does not exist.
        )
    ) else (
        echo Skipping %%f
    )
)

echo Clang-Format completed.
pause
