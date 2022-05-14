import tkinter as tk
import modules


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry('600x400')

        self.myfunction = modules.MyOperations()

        self.main_menu = tk.Menu(window)  # 創一個視窗

        self.file_menu = tk.Menu(self.main_menu, tearoff=0)  #自訂field
        self.main_menu.add_cascade(label='file', menu=self.file_menu)  # 選單第一欄
        self.file_menu.add_command(label="open file", command=self.myfunction.open_file)  # 選單下選項
        self.file_menu.add_command(label='save file', command=self.myfunction.save_file)
        self.file_menu.add_separator()  # 分隔線
        self.file_menu.add_command(label='exit', command=window.quit)

        self.operation_menu = tk.Menu(self.main_menu, tearoff=0)     #下拉的
        self.main_menu.add_cascade(label='function', menu=self.operation_menu)  # 選單第二欄
        self.operation_menu.add_command(label='gray', command=self.myfunction.gray_image)
        self.operation_menu.add_command(label='histogram', command=self.myfunction.img_histogram)
        self.operation_menu.add_command(label='histogram equalization', command=self.myfunction.hist_equa)
        self.operation_menu.add_command(label='ROI', command=self.myfunction.roi_cut)

        self.operation2_menu = tk.Menu(self.main_menu, tearoff=0)    # tearoff=0選單不可獨立 1可
        self.main_menu.add_cascade(label='function2', menu=self.operation2_menu)  # 選單第三欄
        self.operation2_menu.add_command(label='canny edge detector', command=self.myfunction.canny_detector)
        self.operation2_menu.add_command(label='thresholding', command=self.myfunction.thresholding)
        self.operation2_menu.add_command(label='Hough Transform', command=self.myfunction.hough_transform)
        self.operation2_menu.add_command(label='filter', command=self.myfunction.filtering)
        self.operation2_menu.add_command(label='affine', command=self.myfunction.affine)
        self.operation2_menu.add_command(label='perspective', command=self.myfunction.perspective)

        self.operation3_menu = tk.Menu(self.main_menu, tearoff=0)
        self.main_menu.add_cascade(label='function3', menu=self.operation3_menu)
        self.operation3_menu.add_command(label='simple contour', command=self.myfunction.simple_contour)
        self.operation3_menu.add_command(label='find contour', command=self.myfunction.find_contour)
        self.operation3_menu.add_command(label='convex hull', command=self.myfunction.convex_hull)
        self.operation3_menu.add_command(label= 'bounding box', command=self.myfunction.bounding_box)

        self.operation4_menu = tk.Menu(self.main_menu, tearoff=0)
        self.main_menu.add_cascade(label='function4', menu=self.operation4_menu)
        self.operation4_menu.add_command(label='basic morphology', command=self.myfunction.basic_morphology)
        self.operation4_menu.add_command(label='advanced morphology', command=self.myfunction.advanced_morphology)

        self.window.config(menu=self.main_menu)
        self.window.mainloop()      #讓視窗不斷整理

App(tk.Tk(), "window")