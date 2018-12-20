import sys
import os

from cx_Freeze import setup, Executable

dll_path = r'C:\Anaconda\Library\bin'
files = [file for file in os.listdir() if (('.bat' in file) or ('accdb' in file))]
dlls = [os.path.join(dll_path, file) for file in os.listdir() if '.dll' in file] + [r'C:\Anaconda\DLLs\sqlite3.dll']

include_files = files + dlls

build_exe_options = {
    "packages": ["os", "pandas", "pulp", "numpy", "pandas_datareader", "openpyxl"],
    "include_files": include_files}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
# if sys.platform == "win32":
#     base = "Win32GUI"

os.environ['TCL_LIBRARY'] = r'C:\Anaconda\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Anaconda\tcl\tk8.6'
   
setup(  name = "ordinamento",
        version = "0.1",
        options = {"build_exe": build_exe_options},
        executables = [Executable("solveproblem.py", base=base)])