ffmpeg -ss 30 -t 3 -i ..\results\vid1_seg.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 ../hog_finder.gif
