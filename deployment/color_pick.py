'''
Reference: https://blog.csdn.net/dongfuguo/article/details/118704759
'''

import tkinter
import tkinter.filedialog
import tkinter.messagebox
from PIL import ImageGrab, Image, ImageTk

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # if your windows version >= 8.1
except:
    ctypes.windll.user32.SetProcessDPIAware() # win 8.0 or less

class ColorPick:
    def __init__(self,root):
        self.img = ImageGrab.grab()
        screenWidth, screenHeight = self.img.size

        self.top = tkinter.Toplevel(root, width=screenWidth, height=screenHeight)
        self.picked_color = (0,0,0)
        self.top.overrideredirect(True)
        self.tk_image = ImageTk.PhotoImage(self.img)

        self.canvas = tkinter.Canvas(self.top, bg='white', width=screenWidth, height=screenHeight,cursor='target')

        self.canvas.create_image(screenWidth // 2, screenHeight // 2, image=self.tk_image)

        def onLeftButtonDown(event):
            color = self.img.getpixel((event.x, event.y))
            self.picked_color = color
            self.top.destroy()
        self.canvas.bind('<ButtonRelease-1>', onLeftButtonDown)
        self.canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)
    def get_color(self):
        return self.picked_color
