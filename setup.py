import sys
import os
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["os", "pandas", "pulp", "numpy"]}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

os.environ['TCL_LIBRARY'] = r'C:\Anaconda\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Anaconda\tcl\tk8.6'
   
setup(  name = "ordinamento",
        version = "0.1",
        options = {"build_exe": build_exe_options},
        executables = [Executable("solveproblem.py", base=base)])