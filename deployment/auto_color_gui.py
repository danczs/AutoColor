import sys
sys.path.append("..")
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk
from tkinter import filedialog

from tkinter.colorchooser import askcolor
import numpy as np
import os
import time
from color_pick import ColorPick

# using pytorch or onnx deployment
#from autocolor_pytorch_deployment import AutoColorDeployment # pytorch deployment
from autocolor_onnx_deployment import AutoColorDeployment # onnx deployment

IMAGE_SIZE = 448
BASE_SIZE = 224

class CanvasState():
    NORM = 1
    PEN = 2
    ERASER = 3
    def __init__(self):
        super().__init__()
        self.state = CanvasState.NORM
        self.pen_color = '#FF0000'
        self.base_pen_size = IMAGE_SIZE // BASE_SIZE
        self.pen_size = self.base_pen_size * 4
        self.pen_color_array = np.array([255, 0, 0])
        self.drawing_rect = False
        self.rect_scale = 1
        self.rect_x = 0
        self.rect_y = 0
        self.rect_end_x = 0
        self.rect_end_y = 0
        self.canvas_scale = 1

    def get_state(self):
        return self.status

    def set_state(self, state):
        assert state in [CanvasState.NORM, CanvasState.PEN, CanvasState.ERASER]
        self.state = state
        return self.state

    def is_norm(self):
        return self.state == CanvasState.NORM

    def using_pen(self):
        return self.state == CanvasState.PEN

    def using_eraser(self):
        return self.state == CanvasState.ERASER

if __name__ == '__main__':
    #init
    root = tk.Tk()
    root.title('AutoColor')
    root.geometry('1400x780+100+10')
    root.iconbitmap('feather_icon.ico')
    image_path = './example_white.jpg'
    pil_image = Image.open(image_path)

    # input image and input info frame
    input_image_info_frame = tk.Frame(root)
    input_image_info_frame.pack(side='left', anchor=tk.N)

    # input_image_info_frame -> input_image_frame
    input_image_frame = tk.Frame(input_image_info_frame)
    input_image_frame.pack(side='left', anchor=tk.N, padx=10, pady=10)
    image_main_label = tk.Label(input_image_frame, text='待上色图片')
    image_main_label.pack(anchor=tk.W)
    image_main_frame = tk.Frame(input_image_frame)
    image_main_frame.pack()
    canvas_state = CanvasState()
    color_grids = np.zeros((BASE_SIZE, BASE_SIZE, 3), dtype=np.uint8)
    auto_color_model = AutoColorDeployment()
    np.random.seed(0)
    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_main = ImageTk.PhotoImage(pil_image)
    input_image = [[image_main, pil_image, pil_image]]
    output_images = [[pil_image, pil_image]]
    input_image_config = []

    # resize and pad an image
    def resize_and_pad(image, size, pad_color='black',resample=Image.Resampling.BICUBIC):#Resampling.
        w, h = image.size
        if w > h:
            new_w, new_h = size, round(size / w * h)
        else:
            new_w, new_h = round(size / h * w), size
        image_resize = image.resize((new_w, new_h), resample=resample)#Resampling.
        pad_image = Image.new(mode='RGB', size=(size, size), color=pad_color)
        pad_t, pad_l = (size - new_h) // 2, (size - new_w) // 2
        pad_b, pad_r = size - new_h - pad_t, size - new_w - pad_l
        pad_image.paste(image_resize, (pad_l, pad_t))
        return pad_image, (pad_l, pad_r, pad_t, pad_b)

    # adjust the input gray image by add value to pixels
    def scale_gray_image(value):
        tk_img, img, img_ori = input_image.pop()
        image_ori_np = np.array(img_ori,dtype=np.int32)
        image_ori_np = image_ori_np + int(value)
        image_ori_np = np.clip(image_ori_np,0,255)
        image_ori_np = image_ori_np.astype(np.uint8)
        image_ori = Image.fromarray(image_ori_np)
        img,_ = resize_and_pad(image_ori,IMAGE_SIZE * canvas_state.canvas_scale,pad_color='black')
        tk_img = ImageTk.PhotoImage(img)
        input_image.append([tk_img, img, img_ori])
        cv_image = image_canvas.find_withtag('image')
        image_canvas.itemconfig(cv_image[0],image=input_image[0][0])

    #select the input image and convert it to grayscale
    def select_input_img():
        imagedir = filedialog.askopenfilenames()
        if len(imagedir) == 0:
            return
        img_ori = Image.open(imagedir[0])
        img_ori = img_ori.convert('L').convert('RGB')

        while len(input_image_config) > 0:
            input_image_config.pop()
        input_image_config.append([imagedir[0]])
        img, _ = resize_and_pad(img_ori, size=IMAGE_SIZE)
        tk_img = ImageTk.PhotoImage(img)
        while len(input_image) > 0:
            input_image.pop()
        input_image.append([tk_img, img, img_ori])
        image_canvas.delete("all")
        image_canvas.create_image(IMAGE_SIZE // 2, IMAGE_SIZE // 2, image=input_image[0][0], tags=['image'])
        color_grids[:, :, :] = 0
        gray_scale.set(0)

    #scale bar for ajusting the input gray image
    gray_scale = tk.Scale(image_main_frame,
                          from_=-255,
                          to=255,
                          resolution=3,
                          length=IMAGE_SIZE,
                          sliderlength=20,
                          #showvalue=False,
                          command=scale_gray_image)
    gray_scale.pack(side='left',anchor=tk.W)
    gray_scale.set(0)
    image_canvas = tk.Canvas(image_main_frame, bg="white", width=IMAGE_SIZE, height=IMAGE_SIZE)
    image_canvas.pack(side='left')
    image_canvas.config(scrollregion=(0,0,IMAGE_SIZE,IMAGE_SIZE))
    scroll_x_frame = tk.Frame(input_image_frame)
    scroll_x_frame.pack(side=tk.TOP, fill=tk.X)

    #scale the canvas size
    def canvas_scaling():
        canvas_scale = canvas_state.canvas_scale
        #print(hbar.get(),vbar.get())
        if  canvas_state.canvas_scale == 1:
            canvas_state.canvas_scale = 2
            canvas_scaling_button.config(text = '缩小图片')
        else:
            canvas_state.canvas_scale = 1
            canvas_scaling_button.config(text = '放大图片')
        real_size = IMAGE_SIZE * canvas_state.canvas_scale
        image_canvas.config(
            scrollregion=(0, 0,real_size , real_size)
        )
        tk,_,img_ori =input_image[0]
        img, _ = resize_and_pad(img_ori, size=real_size)
        tk_img = ImageTk.PhotoImage(img)
        input_image[0] = [tk_img,img,img_ori]
        image_canvas.create_image(real_size // 2, real_size // 2, image=input_image[0][0], tags=['image'])

        #adjust the color info accordingly
        all_rect_items = image_canvas.find_withtag('rect')
        for i in range(len(all_rect_items)):
            x0, y0, x1, y1 = image_canvas.coords(all_rect_items[i])
            factor = canvas_state.canvas_scale if canvas_state.canvas_scale==2 else 0.5
            x0, y0 = int((x0 - 1) * factor) , int((y0 - 1) * factor)
            x1, y1 = int((x1 + 1) * factor), int((y1 + 1) * factor)
            out_line = image_canvas.itemcget(all_rect_items[i], 'outline')
            image_canvas.create_rectangle(x0 + 1, y0 + 1, x1 - 1, y1 - 1,
                                          outline= out_line,
                                          fill=out_line,
                                          tags=['rect'])
            image_canvas.delete(all_rect_items[i])

        # adjust the selected local input accordingly
        local_input_items = image_canvas.find_withtag('local_input')
        for i in range(len(local_input_items)):
            x0, y0, x1, y1 = image_canvas.coords(local_input_items[i])
            factor = canvas_state.canvas_scale if canvas_state.canvas_scale == 2 else 0.5
            x0, y0 = int((x0 - 1) * factor), int((y0 - 1) * factor)
            x1, y1 = int((x1 + 1) * factor), int((y1 + 1) * factor)
            out_line = image_canvas.itemcget(local_input_items[i], 'outline')
            image_canvas.create_rectangle(x0, y0, x1, y1,
                                          outline=out_line,
                                          #fill=out_line,
                                          tags=['local_input'])
            image_canvas.delete(local_input_items[i])

    canvas_scaling_button = tk.Button(scroll_x_frame, text='放大图片', command=canvas_scaling)
    canvas_scaling_button.pack(side=tk.LEFT)

    #config the scrollbars
    hbar = tk.Scrollbar(scroll_x_frame, orient=tk.HORIZONTAL)
    hbar.pack(side=tk.TOP, fill=tk.X, anchor=tk.E)
    hbar.config(command=image_canvas.xview)
    vbar = tk.Scrollbar(image_main_frame, orient=tk.VERTICAL)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    vbar.config(command=image_canvas.yview)

    image_canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    input_image_button = tk.Button(input_image_frame, text='选择输入图片')
    input_image_button.pack(pady=3)
    input_image_button.config(command=select_input_img)

    #input color info frame
    color_info_frame = tk.LabelFrame(input_image_frame, text='参考颜色', padx=5, pady=5)
    color_info_frame.pack(fill=tk.X)

    #pen and eraser
    selet_frame = tk.Frame(color_info_frame)
    selet_frame.pack(pady=5, fill=tk.X)
    pen_button = tk.Button(selet_frame, text='使用画笔')
    pen_button.pack(side='left')
    color_canvas = tk.Canvas(selet_frame,
                             bg='red',
                             height=20,
                             width=20)
    color_canvas.pack(side='left', padx=10)
    color_button = tk.Button(selet_frame, text='选择画笔颜色')
    color_button.pack(side='left',padx=2)

    color_pick_button = tk.Button(selet_frame, text='屏幕取色')
    color_pick_button.pack(side='left')

    pen_size_frame = tk.Frame(selet_frame)
    pen_size_frame.pack(pady=5, fill=tk.X)
    pen_size = ['4', '8', '16']#'1', '2',
    combobox = ttk.Combobox(pen_size_frame, width=11)
    combobox.pack(side='right',padx=5)
    combobox['value'] = pen_size
    combobox.current(0)

    pen_size_label = tk.Label(pen_size_frame, text='画笔尺寸:')
    pen_size_label.pack(side='right')

    eraser_and_clear_frame = tk.Frame(color_info_frame)
    eraser_and_clear_frame.pack(fill=tk.X,side='top',pady=2)
    eraser_button = tk.Button(eraser_and_clear_frame, text='使用橡皮')
    eraser_button.pack(side='left')
    clear_button = tk.Button(eraser_and_clear_frame, text='清空颜色')
    clear_button.pack(side='right')

    def set_color_canvas(i):
        i = int(i)
        color_canvas.config(bg = color_rgb_list[i][1])
        canvas_state.pen_color = color_rgb_list[i][1]
        canvas_state.pen_color_array = color_rgb_list[i][0]
        color_use_time[i] = time.time()

    def set_color_btn(color):
        select_btn_index = 0
        min_time = color_use_time[0]
        for i, use_time in enumerate(color_use_time):
            if use_time < min_time:
                select_btn_index = i
                min_time = use_time
        color_use_time[select_btn_index] = time.time()

        color_rgb_list[select_btn_index] = color
        color_button_list[select_btn_index].config(bg=color[1])
        set_color_canvas(select_btn_index)

    def rgb2hexstr(color):
        rgb = [int(v) for v in color]
        rgb_hex = [hex(v)[2:] for v in rgb]
        rgb_hex = [ ('0'+v if len(v)<2 else v ) for v in rgb_hex]
        return '#' + ''.join(rgb_hex)

    def hexstr2rgb(color):
        assert len(color) == 7 and color[0]=='#'
        color = color.lower()
        r,g,b = color[1:3],color[3:5],color[5:7]
        return [int(r,16),int(g,16),int(b,16)]

    def pick_color_from_screen():
        w = ColorPick(root)
        color_pick_button.wait_window(w.top)
        root.state('normal')
        color = w.get_color()
        color_str = rgb2hexstr(color)
        set_color_btn([color, color_str])

    #color pick from screen
    color_pick_button.config(command=pick_color_from_screen)

    #config the alternate color list
    #using the time stamps to keep the recently used color
    color_button_frame = tk.Frame(color_info_frame)
    color_button_frame.pack(fill=tk.X,side='top')
    color_num = 13
    color_button_list = []
    color_rgb_list = []
    color_use_time=[0]*color_num
    for _ in range(color_num):
        color_int = np.random.randint(255,size=3)
        color_rgb_list.append([color_int,rgb2hexstr(color_int)])
    for i in list(range(color_num)):
        fn = lambda a=i: set_color_canvas(a)
        color_btn = tk.Button(color_info_frame,
                              height=1,
                              width=3,
                              bd=0,
                              bg=color_rgb_list[i][1],
                              command=fn,
                              )
        color_btn.pack(side='left',padx=3)
        color_button_list.append(color_btn)

    #config the local input
    rect_input_button = tk.Button(input_image_frame, text='选择上色区域')
    rect_input_button.pack(pady=10)

    def select_rect_input():
        drawing_rect = canvas_state.drawing_rect
        if drawing_rect:
            canvas_state.drawing_rect = False
            rect_input_button.config(text='选择上色区域')
            rect_item = image_canvas.find_withtag('local_input')
            image_canvas.delete(rect_item)
        else:
            canvas_state.drawing_rect = True
            rect_input_button.config(text='清除选择框')
            if canvas_state.using_pen():
                pen_button.config(text='使用画笔')
            if canvas_state.using_eraser():
                eraser_button.config(text='使用橡皮')
            canvas_state.set_state(CanvasState.NORM)
            image_canvas.config(cursor='arrow')

    rect_input_button.config(command=select_rect_input)

    # input_image_info_frame -> middle_info_frame
    middle_info_frame = tk.Frame(input_image_info_frame, padx=20)
    middle_info_frame.pack(side='right', anchor=tk.N, pady=10)
    image_info_frame = tk.LabelFrame(middle_info_frame, text='参考图片', padx=5, pady=5)
    image_info_frame.pack()
    image_choose_button = tk.Button(image_info_frame, text='选择参考图片')
    image_choose_button.pack(anchor=tk.W,pady=10)
    info_image = pil_image.resize((IMAGE_SIZE // 2, IMAGE_SIZE // 2))
    info_image = ImageTk.PhotoImage(info_image)
    image_info_label = tk.Label(image_info_frame, image=info_image)
    image_info_label.pack(side='left', fill=tk.X,pady=10)
    info_images = []
    tk_info_images = []

    #selet the input info images
    def selet_info_image():
        MAX_INFO_NUM = 16
        pad_pixel = 4
        imagedir = filedialog.askopenfilenames()
        if len(imagedir) == 0:
            return
        image_num = len(imagedir)
        if image_num > MAX_INFO_NUM:
            image_num = MAX_INFO_NUM
        while len(info_images) > 0:
            info_images.pop()
        info_grid = int(np.sqrt(image_num))
        info_grid = info_grid if info_grid*info_grid == image_num else info_grid + 1
        if info_grid == 1:
            pad_pixel = 0
        image_show_size = (BASE_SIZE - pad_pixel* (info_grid-1) ) // info_grid
        combine_show_image = Image.new(mode='RGB',size=(BASE_SIZE,BASE_SIZE),color="#ffffff")
        for i in range(info_grid):
            for j in range(info_grid):
                image_index = i*info_grid + j
                if image_index <= image_num - 1:
                    pad_i = i * (pad_pixel + image_show_size )
                    pad_j = j * (pad_pixel + image_show_size )
                    img_ori = Image.open(imagedir[image_index])
                    img_ori = img_ori.convert('RGB')
                    img_info, _ = resize_and_pad(img_ori, size=BASE_SIZE)
                    info_images.append(img_info)
                    img_show, _ = resize_and_pad(img_info,size=image_show_size)
                    combine_show_image.paste(img_show,(pad_j,pad_i))

        tk_img = ImageTk.PhotoImage(combine_show_image)
        while len(tk_info_images)>0:
            tk_info_images.pop()
        #keep the pointer of tk_img to avoid recycling
        tk_info_images.append(tk_img)
        image_info_label.config(image=tk_img)

    image_choose_button.config(command=selet_info_image)

    # input_image_info_frame -> middle_info_frame -> text info frame
    text_info_frame = tk.LabelFrame(middle_info_frame, text='参考文本', padx=15,pady=10)
    text_info_frame.pack(fill=tk.X,pady=10)
    text = tk.Text(text_info_frame, width=10, height=10, undo=True, autoseparators=False)
    text.pack(fill=tk.X)

    # clear the canvas
    def clear_canvas():
        ori_img = input_image[0][1]
        tk_image = ImageTk.PhotoImage(ori_img)
        input_image[0][0] = tk_image
        image_canvas.delete("all")
        real_size = IMAGE_SIZE * canvas_state.canvas_scale
        image_canvas.create_image(real_size // 2, real_size // 2, image=input_image[0][0],tags=['image'])
        color_grids[:, :, :] = 0

    # select or unselect the pen
    def click_pen():
        if canvas_state.using_pen():
            canvas_state.set_state(CanvasState.NORM)
            pen_button.config(text='使用画笔')
            image_canvas.config(cursor='arrow')
        else:
            if canvas_state.using_eraser():
                eraser_button.config(text='使用橡皮')
            canvas_state.set_state(CanvasState.PEN)
            pen_button.config(text='取消画笔')
            image_canvas.config(cursor='pencil')

    #select or unselect the eraser
    def click_eraser():
        if canvas_state.using_eraser():
            canvas_state.set_state(CanvasState.NORM)
            eraser_button.config(text='使用橡皮')
            image_canvas.config(cursor='arrow')

        else:
            if canvas_state.using_pen():
                pen_button.config(text='使用画笔')

            canvas_state.set_state(CanvasState.ERASER)
            eraser_button.config(text='取消橡皮')
            image_canvas.config(cursor='tcross')

    def selet_pen_size(event):
        pen_size = combobox.get()
        canvas_state.pen_size = canvas_state.base_pen_size * int(pen_size)

    #select pen color
    def select_color():
        color = askcolor(title="颜色选择框", color="red")
        if color[1] is None:
            return
        color_canvas.config(bg=color[1])
        canvas_state.pen_color = color[1]
        canvas_state.pen_color_array = np.array(color[0])
        set_color_btn(color)

    color_button.config(command=select_color)
    clear_button.config(command=clear_canvas)
    combobox.bind('<<ComboboxSelected>>', selet_pen_size)
    pen_button.config(command=click_pen)
    eraser_button.config(command=click_eraser)
    last_x = tk.IntVar(value=0)
    last_y = tk.IntVar(value=0)

    # canvas event: LeftButtonDown
    def onLeftButtonDown(event):
        scroll_x = hbar.get()
        scroll_y = vbar.get()
        scroll_x = round(scroll_x[0] * IMAGE_SIZE * canvas_state.canvas_scale)
        scroll_y = round(scroll_y[0] * IMAGE_SIZE * canvas_state.canvas_scale)
        event.x = event.x + scroll_x
        event.y = event.y + scroll_y
        pen_size = canvas_state.pen_size
        if canvas_state.is_norm():
            if canvas_state.is_norm():
                if canvas_state.drawing_rect:
                    canvas_state.rect_x = event.x
                    canvas_state.rect_y = event.y
                    canvas_state.rect_scale = canvas_state.canvas_scale
            return
        elif canvas_state.using_pen():
            x = event.x
            y = event.y
            x = x // pen_size * pen_size
            y = y // pen_size * pen_size

            image_canvas.create_rectangle(x + 1, y + 1, x + pen_size - 1, y + pen_size - 1,
                                          outline=canvas_state.pen_color,
                                          fill=canvas_state.pen_color,tags=['rect'])

            x1 = x // canvas_state.base_pen_size // canvas_state.canvas_scale
            y1 = y //canvas_state.base_pen_size // canvas_state.canvas_scale
            step = pen_size // canvas_state.base_pen_size // canvas_state.canvas_scale
            for i in range(3):
                color_grids[x1:x1 + step, y1:y1 + step, i] = int(canvas_state.pen_color_array[i])

            last_x.set(x)
            last_y.set(y)
        elif canvas_state.using_eraser():
            pen_size = min(pen_size,4)
            x = event.x
            y = event.y
            x = x // pen_size * pen_size
            y = y // pen_size * pen_size

            items = image_canvas.find_overlapping(x + 1, y + 1, x + pen_size - 1, y + pen_size - 1)
            for i in range(len(items)):
                tags = image_canvas.itemcget(items[i], 'tags')

                if 'rect' in tags:
                    x0,y0,x1,y1 = image_canvas.coords(items[i])
                    x0, y0 = int(x0 - 1) // canvas_state.base_pen_size, int(y0 - 1)//canvas_state.base_pen_size
                    x1, y1 = int(x1 + 1) // canvas_state.base_pen_size,  int(y1 + 1)//canvas_state.base_pen_size
                    color_grids[x0:x1, y0:y1, :] = 0
                    image_canvas.delete(items[i])

    #canvas event: LeftButtonMove
    def onLeftButtonMove(event):
        scroll_x = hbar.get()
        scroll_y = vbar.get()
        scroll_x = round(scroll_x[0] * IMAGE_SIZE * canvas_state.canvas_scale)
        scroll_y = round(scroll_y[0] * IMAGE_SIZE * canvas_state.canvas_scale)
        event.x = event.x + scroll_x
        event.y = event.y + scroll_y
        pen_size = canvas_state.pen_size

        if canvas_state.is_norm():
            if canvas_state.drawing_rect:
                rect_item = image_canvas.find_withtag('local_input')
                #print(rect_item)
                image_canvas.delete(rect_item)
                image_canvas.create_rectangle(canvas_state.rect_x,canvas_state.rect_y,event.x,event.y,
                                              tag='local_input',outline='red')
                canvas_state.rect_end_x = event.x
                canvas_state.rect_end_y = event.y
            return
        elif canvas_state.using_pen():
            x = event.x
            y = event.y
            x = x // pen_size * pen_size
            y = y // pen_size * pen_size
            if x != last_x.get() or y != last_y.get():
                #print('moving create',x,y,last_x.get(),last_y.get(), abs(x - last_x.get()) + abs(y - last_x.get() ))
                image_canvas.create_rectangle(x + 1, y + 1, x + pen_size - 1, y + pen_size - 1,
                                              outline=canvas_state.pen_color,
                                              fill=canvas_state.pen_color,
                                              tags=['rect'])
                last_x.set(x)
                last_y.set(y)

                x1 = x // canvas_state.base_pen_size // canvas_state.canvas_scale
                y1 = y // canvas_state.base_pen_size // canvas_state.canvas_scale
                step = pen_size // canvas_state.base_pen_size // canvas_state.canvas_scale
                for i in range(3):
                    color_grids[x1:x1 + step, y1:y1 + step, i] = int(canvas_state.pen_color_array[i])


        elif canvas_state.using_eraser():
            pen_size = max(pen_size,4)
            x = event.x
            y = event.y
            x = x // pen_size * pen_size
            y = y // pen_size * pen_size
            if x != last_x.get() or y != last_x.get():
                items = image_canvas.find_overlapping(x + 1, y + 1, x + pen_size - 1, y + pen_size - 1)
                # print(items)
                for i in range(len(items)):
                    tags = image_canvas.itemcget(items[i], 'tags')
                    if 'rect' in tags:
                        x0, y0, x1, y1 = image_canvas.coords(items[i])
                        x0, y0 = int(x0 - 1) // canvas_state.base_pen_size, int(y0 - 1) // canvas_state.base_pen_size
                        x1, y1 = int(x1 + 1) // canvas_state.base_pen_size, int(y1 + 1) // canvas_state.base_pen_size
                        color_grids[x0:x1, y0:y1, :] = 0
                        image_canvas.delete(items[i])

                last_x.set(x)
                last_y.set(y)

    image_canvas.bind('<Button-1>', onLeftButtonDown)
    image_canvas.bind('<B1-Motion>', onLeftButtonMove)

    #utilize gpu or cpu
    use_gpu = tk.IntVar()
    def set_auto_color_device():
        auto_color_model.set_device(use_gpu.get())
    gpu_device = tk.Checkbutton(middle_info_frame, text="使用gpu", variable=use_gpu, command=set_auto_color_device)
    gpu_device.pack(pady=15)

    # generate colorful image
    generate_button = tk.Button(middle_info_frame, text='执行自动上色', pady=15)
    generate_button.pack(fill=tk.X, padx=5, pady=0)
    generate_button.config(fg='red')

    # get different resolution inputs for the model
    def get_diff_res_inputs(image_size, target_size,image):
        input_image_model = []

        while image_size < target_size:
            temp, pad = resize_and_pad(image, image_size, pad_color='black')
            input_image_model.append(temp)
            image_size = image_size * 2
        temp, pad = resize_and_pad(image, target_size, pad_color='black')
        input_image_model.append(temp)
        return input_image_model, pad

    # get scaled gray image
    def get_scaled_gray_image(value,image):
        if value == 0:
            return image
        image = np.array(image,dtype=np.int32)
        image = image + int(value)
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        return image

    #colorize the input image with the AI model
    def auto_color():
        scale_value = gray_scale.get()
        if canvas_state.drawing_rect:
            #local rect input
            input_image_ori = input_image[0][2]
            input_image_ori = get_scaled_gray_image(scale_value, input_image_ori)
            w, h = input_image_ori.size
            new_size = max(w,h)
            temp, pad_ori = resize_and_pad(input_image_ori, new_size, pad_color='black')
            crop_x = round(canvas_state.rect_x // canvas_state.rect_scale  * new_size / IMAGE_SIZE)
            crop_y = round(canvas_state.rect_y // canvas_state.rect_scale * new_size / IMAGE_SIZE)
            crop_end_x = round(canvas_state.rect_end_x // canvas_state.rect_scale * new_size / IMAGE_SIZE)
            crop_end_y = round(canvas_state.rect_end_y // canvas_state.rect_scale* new_size / IMAGE_SIZE)
            crop_image = temp.crop((crop_x,crop_y,crop_end_x,crop_end_y))
            target_size = max([crop_end_y - crop_y, crop_end_x - crop_x])
            target_size = max([target_size,BASE_SIZE])
            diff_res_inputs,pad = get_diff_res_inputs(BASE_SIZE,target_size,crop_image)
        else:
            # full image input
            input_image_ori = input_image[0][2]
            input_image_ori = get_scaled_gray_image(scale_value, input_image_ori)
            w, h = input_image_ori.size
            target_size = max([w, h])
            image_size = BASE_SIZE

            diff_res_inputs,pad_ori = get_diff_res_inputs(image_size,target_size,input_image_ori)
        if len(input_image_config[0])<2:
            input_image_config[0].append(pad_ori)

        image_info = info_images

        input_text = text.get('0.0', 'end')
        input_text = input_text.strip()
        input_text = None if len(input_text) == 0 else input_text

        #local rect input
        if canvas_state.drawing_rect:
            color_info = color_grids.repeat(canvas_state.base_pen_size, axis=0).repeat(canvas_state.base_pen_size, axis=1)

            x0 = canvas_state.rect_x // canvas_state.rect_scale
            x1 = canvas_state.rect_end_x // canvas_state.rect_scale
            y0 = canvas_state.rect_y // canvas_state.rect_scale
            y1 = canvas_state.rect_end_y // canvas_state.rect_scale
            color_info = color_info[x0:x1, y0:y1,:]
            color_info = np.transpose(color_info, (1, 0, 2))
            color_info = Image.fromarray(color_info)
            color_info, _ = resize_and_pad(color_info, BASE_SIZE, pad_color='black', resample=Image.Resampling.NEAREST)
            output_image = auto_color_model.autocolor_forward(diff_res_inputs, image_info, input_text, color_info)

            if len(output_images)>0:
                _,output_image_ori = output_images.pop()
            else:
                output_image_ori,pad_ori = resize_and_pad(input_image_ori, max(input_image_ori.size), pad_color='black')
            pad_l, pad_r, pad_t, pad_b = pad
            output_image = output_image.crop( (pad_l,pad_t,target_size - pad_r, target_size - pad_b))
            output_image = output_image.resize((crop_end_x-crop_x,crop_end_y - crop_y))
            output_image_ori.paste(output_image,(crop_x,crop_y))
            output_image_show = output_image_ori.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC)
            tk_output_image = ImageTk.PhotoImage(output_image_show)
            while len(output_images) > 0:
                output_images.pop()
            output_images.append([tk_output_image, output_image_ori])
            right_image_label.config(image=tk_output_image)
        else:
            # full image input
            color_info = color_grids
            color_info = np.transpose(color_info, (1, 0, 2))
            output_image = auto_color_model.autocolor_forward(diff_res_inputs, image_info, input_text, color_info)
            output_image_resize = output_image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC)
            # output_image.show()
            while len(output_images) > 0:
                output_images.pop()

            tk_output_image = ImageTk.PhotoImage(output_image_resize)
            output_images.append([tk_output_image, output_image])
            right_image_label.config(image=tk_output_image)

    generate_button.config(command=auto_color)

    # right_output_frame
    right_output_frame = tk.Frame(root)
    right_output_frame.pack(side='left', anchor=tk.N, pady=10)
    right_output_label = tk.Label(right_output_frame, text='已上色图片')
    right_output_label.pack(anchor=tk.W)
    right_image_label = tk.Label(right_output_frame, image=image_main)
    right_image_label.pack()

    fast_save_button = tk.Button(right_output_frame, text='快速保存')
    fast_save_button.pack(side='right', pady=5)

    output_save_button = tk.Button(right_output_frame, text='保存输出图片')
    output_save_button.pack(side='left',pady=5)


    output_show_button = tk.Button(right_output_frame, text='展示图片')
    output_show_button.pack(pady=5)

    def save_output_image():
        file_path = filedialog.asksaveasfilename(title=u'保存文件')
        if len(file_path) == 0:
            return
        _, image = output_images[0]
        path, pad = input_image_config[0]
        pad_l, pad_r, pad_t, pad_b = pad
        w, h = image.size
        image = image.crop((pad_l, pad_t, w - pad_r, h - pad_b))
        image.save(file_path)


    def fast_save():
        _, image = output_images[0]
        path, pad = input_image_config[0]
        pad_l, pad_r, pad_t, pad_b = pad
        w, h = image.size
        new_file_name = os.path.join(os.path.dirname(path), 'autocolor_' + os.path.basename(path))
        image = image.crop((pad_l, pad_t, w - pad_r, h - pad_b))
        image.save(new_file_name)

    def show_output_image():
        _, image = output_images[0]
        image.show()
        # color_info = np.transpose(color_grids, (1, 0, 2))
        # Image.fromarray(color_info).show()

    output_save_button.config(command=save_output_image)
    output_show_button.config(command=show_output_image)
    fast_save_button.config(command=fast_save)

    root.mainloop()


