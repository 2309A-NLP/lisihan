@echo off
cd /d C:\
if exist C:\Users\freedom\Desktop\zhaogu rmdir C:\Users\freedom\Desktop\zhaogu
mklink /J C:\Users\freedom\Desktop\zhaogu "C:\Users\freedom\Desktop\招股书问答智能体"
echo Done
