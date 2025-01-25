import sys
import os.path
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk
from tkinter import filedialog
try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True
import Home  # Assuming Home module is imported for Home.vp_start_gui()


def start_gui():
    '''Starting point when module is the main routine.'''
    global val, main_window, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    root = tk.Tk()
    root.mainloop()


main_window = None


def create_main_window(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global main_window, window, rt
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    rt = root
    return window, main_window


def destroy_main_window():
    global window
    window.destroy()
    window = None

Home.start_gui()
start_gui()


