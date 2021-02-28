@REM echo off

set video_dir=%1

if "%video_dir%"=="" (
    set video_dir=videos\train
)

SETLOCAL ENABLEDELAYEDEXPANSION
for %%f in (%video_dir%\*.mp4) do (
    echo %%~f
    set new_path=segmentation\data\%%~nf
    mkdir !new_path!
    ffmpeg.exe -i %%~f -r 0.2 !new_path!\%%d.png
)