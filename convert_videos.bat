echo off

SETLOCAL ENABLEDELAYEDEXPANSION
for %%f in (videos\*.mp4) do (
    echo %%~f
    set new_path=data\%%~nf
    mkdir !new_path!
    ffmpeg.exe -i %%~f -r 0.2 !new_path!\%%d.png
)