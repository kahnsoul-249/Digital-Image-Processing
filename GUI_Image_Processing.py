# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:14:06 2026

@author: 22110068
Huynh Minh Tai
"""
import cv2
import numpy as np
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class ImageProcessingApp:
    def __init__(self, root):
        """Khởi tạo ứng dụng GUI"""
        self.root = root
        self.root.title("Digital Image Processing Application")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        self.original_image = None
        self.processed_image = None
        self.current_image = None  
        
        self.image_loaded = False
        
        self.create_widgets()
        
    def create_widgets(self):
        """Tạo các widget cho GUI"""
        
        title_frame = Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=X, side=TOP)
        
        title_label = Label(title_frame, 
                           text="DIGITAL IMAGE PROCESSING APPLICATION",
                           font=('Arial', 18, 'bold'),
                           bg='#2c3e50', fg='white')
        title_label.pack(pady=15)
        
        button_frame_top = Frame(self.root, bg='#ecf0f1', height=70)
        button_frame_top.pack(fill=X, pady=(0, 5))
        
        Label(button_frame_top, text="ĐIỀU KHIỂN:", 
              font=('Arial', 10, 'bold'), bg='#ecf0f1').pack(side=LEFT, padx=20)
        
        Button(button_frame_top, text="📁 Chọn ảnh", command=self.load_image,
               bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
               width=12, height=1, relief=RAISED, bd=3).pack(side=LEFT, padx=5)
        
        Button(button_frame_top, text="🔄 Reset", command=self.reset_image,
               bg='#95a5a6', fg='white', font=('Arial', 12, 'bold'),
               width=10, height=1, relief=RAISED, bd=3).pack(side=LEFT, padx=5)
        
        Button(button_frame_top, text="💾 Lưu file", command=self.save_image,
               bg='#f39c12', fg='white', font=('Arial', 12, 'bold'),
               width=10, height=1, relief=RAISED, bd=3).pack(side=LEFT, padx=5)
        
        Button(button_frame_top, text="❌ Close", command=self.root.quit,
               bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
               width=10, height=1, relief=RAISED, bd=3).pack(side=RIGHT, padx=20)
        
        main_frame = Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        image_frame = Frame(main_frame, bg='white', relief=RIDGE, bd=2)
        image_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        Label(image_frame, text="Original Image", 
              font=('Arial', 12, 'bold'), bg='white').pack(pady=5)
        
        self.original_canvas = Canvas(image_frame, width=500, height=350, 
                                     bg='#ecf0f1', relief=SUNKEN, bd=2)
        self.original_canvas.pack(padx=10, pady=5)
        
        Label(image_frame, text="Processed Image", 
              font=('Arial', 12, 'bold'), bg='white').pack(pady=5)
        
        self.processed_canvas = Canvas(image_frame, width=500, height=350, 
                                       bg='#ecf0f1', relief=SUNKEN, bd=2)
        self.processed_canvas.pack(padx=10, pady=5)
        
        control_frame = Frame(main_frame, bg='white', relief=RIDGE, bd=2, width=450)
        control_frame.pack(side=RIGHT, fill=Y)
        control_frame.pack_propagate(False)
        
        canvas = Canvas(control_frame, bg='white')
        scrollbar = Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.create_section_header(scrollable_frame, "Negative Image", '#e74c3c')
        neg_frame = self.create_section_frame(scrollable_frame, '#e74c3c')
        
        Button(neg_frame, text="Apply Negative", 
               command=self.apply_negative,
               bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
               width=30).pack(pady=5)
        
        self.create_section_header(scrollable_frame, "Biến đổi Log", '#3498db')
        log_frame = self.create_section_frame(scrollable_frame, '#3498db')
        
        Label(log_frame, text="Hệ số C:", bg='white').pack()
        self.log_c = Scale(log_frame, from_=0.1, to=10, resolution=0.1, 
                          orient=HORIZONTAL, length=350,
                          command=lambda x: self.apply_log())
        self.log_c.set(1.0)
        self.log_c.pack()
        
        self.create_section_header(scrollable_frame, "Biến đổi Piecewise-Linear", '#9b59b6')
        piece_frame = self.create_section_frame(scrollable_frame, '#9b59b6')
        
        Label(piece_frame, text="Hệ số Cao:", bg='white').pack()
        self.piece_high = Scale(piece_frame, from_=0, to=255, resolution=1,
                               orient=HORIZONTAL, length=350,
                               command=lambda x: self.apply_piecewise())
        self.piece_high.set(150)
        self.piece_high.pack()
        
        Label(piece_frame, text="Hệ số Thấp:", bg='white').pack()
        self.piece_low = Scale(piece_frame, from_=0, to=255, resolution=1,
                              orient=HORIZONTAL, length=350,
                              command=lambda x: self.apply_piecewise())
        self.piece_low.set(50)
        self.piece_low.pack()
        
        self.create_section_header(scrollable_frame, "Biến đổi Gamma", '#e67e22')
        gamma_frame = self.create_section_frame(scrollable_frame, '#e67e22')
        
        Label(gamma_frame, text="Hệ số C:", bg='white').pack()
        self.gamma_c = Scale(gamma_frame, from_=0.1, to=5, resolution=0.1,
                            orient=HORIZONTAL, length=350,
                            command=lambda x: self.apply_gamma())
        self.gamma_c.set(1.0)
        self.gamma_c.pack()
        
        Label(gamma_frame, text="Gamma:", bg='white').pack()
        self.gamma_val = Scale(gamma_frame, from_=0.1, to=5, resolution=0.1,
                              orient=HORIZONTAL, length=350,
                              command=lambda x: self.apply_gamma())
        self.gamma_val.set(1.0)
        self.gamma_val.pack()
        
        self.create_section_header(scrollable_frame, "Làm trơn ảnh (lọc trung bình)", '#e91e63')
        smooth_frame = self.create_section_frame(scrollable_frame, '#e91e63')
        
        Label(smooth_frame, text="Kích thước lọc:", bg='white').pack()
        self.smooth_size = Scale(smooth_frame, from_=3, to=31, resolution=2,
                                orient=HORIZONTAL, length=350,
                                command=lambda x: self.apply_smoothing())
        self.smooth_size.set(5)
        self.smooth_size.pack()
        
        self.create_section_header(scrollable_frame, "Làm trơn ảnh (lọc Gauss)", '#ff9800')
        gauss_frame = self.create_section_frame(scrollable_frame, '#ff9800')
        
        Label(gauss_frame, text="Kích thước lọc:", bg='white').pack()
        self.gauss_size = Scale(gauss_frame, from_=3, to=31, resolution=2,
                               orient=HORIZONTAL, length=350,
                               command=lambda x: self.apply_gaussian())
        self.gauss_size.set(5)
        self.gauss_size.pack()
        
        Label(gauss_frame, text="Hệ số Sigma:", bg='white').pack()
        self.gauss_sigma = Scale(gauss_frame, from_=0.1, to=10, resolution=0.1,
                                orient=HORIZONTAL, length=350,
                                command=lambda x: self.apply_gaussian())
        self.gauss_sigma.set(1.0)
        self.gauss_sigma.pack()
        
        self.create_section_header(scrollable_frame, "Làm trơn ảnh (lọc trung vị)", '#e91e63')
        median_frame = self.create_section_frame(scrollable_frame, '#e91e63')
        
        Label(median_frame, text="Kích thước lọc:", bg='white').pack()
        self.median_size = Scale(median_frame, from_=3, to=31, resolution=2,
                                orient=HORIZONTAL, length=350,
                                command=lambda x: self.apply_median())
        self.median_size.set(5)
        self.median_size.pack()

        self.create_section_header(scrollable_frame, "Cân bằng sáng dùng Histogram", '#ffc107')
        hist_frame = self.create_section_frame(scrollable_frame, '#ffc107')
        
        Label(hist_frame, text="Giá trị (clip limit):", bg='white').pack()
        self.hist_clip = Scale(hist_frame, from_=1, to=10, resolution=0.5,
                              orient=HORIZONTAL, length=350,
                              command=lambda x: self.apply_histogram_eq())
        self.hist_clip.set(2.0)
        self.hist_clip.pack()
        
        Button(hist_frame, text="Apply Histogram Equalization",
               command=self.apply_histogram_eq,
               bg='#ffc107', fg='black', font=('Arial', 10, 'bold'),
               width=30).pack(pady=5)
    
    def create_section_header(self, parent, text, color):
        """Tạo header cho mỗi section"""
        header = Label(parent, text=text, bg=color, fg='white',
                      font=('Arial', 11, 'bold'), pady=8)
        header.pack(fill=X, pady=(10, 0))
    
    def create_section_frame(self, parent, color):
        """Tạo frame cho mỗi section"""
        frame = Frame(parent, bg='white', relief=GROOVE, bd=2)
        frame.pack(fill=X, padx=10, pady=(0, 5))
        return frame
    
    
    def load_image(self):
        """Chọn và load ảnh từ file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.current_image = self.original_image.copy()
            
            if self.original_image is not None:
                self.display_image(self.original_image, self.original_canvas)
                self.processed_canvas.delete("all")
                self.image_loaded = True
                messagebox.showinfo("Thành công", "Đã load ảnh thành công!")
            else:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
    
    def display_image(self, cv_image, canvas):
        """Hiển thị ảnh OpenCV lên canvas Tkinter"""
        if cv_image is None:
            return
        
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 500
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else 350
        
        h, w = rgb_image.shape[:2]
        scale = min(canvas_width/w, canvas_height/h)
        new_w, new_h = int(w*scale*0.95), int(h*scale*0.95)
        
        resized = cv2.resize(rgb_image, (new_w, new_h))
        
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Hiển thị lên canvas
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, 
                          image=photo, anchor=CENTER)
        canvas.image = photo  
    
    def apply_negative(self):
        """Biến đổi Negative (Ảnh âm bản)"""
        if not self.image_loaded:
            return 
        self.processed_image = 255 - self.current_image
        self.display_image(self.processed_image, self.processed_canvas)
    
    def apply_log(self):
        """Biến đổi Log"""
        if not self.image_loaded:
            return
        
        c = self.log_c.get()
        
        img_float = self.current_image.astype(np.float32)
        log_image = c * np.log1p(img_float)
        
        log_image = np.clip(log_image, 0, 255)
        self.processed_image = log_image.astype(np.uint8)
        
        self.display_image(self.processed_image, self.processed_canvas)
    
    def apply_piecewise(self):
        """Biến đổi Piecewise-Linear (Contrast Stretching)"""
        if not self.image_loaded:
            return
        
        low = self.piece_low.get()
        high = self.piece_high.get()
        
        img_float = self.current_image.astype(np.float32)
        
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if i < low:
                lut[i] = int(i * (low / 256))
            elif i > high:
                lut[i] = int((i - high) * ((255 - high) / (255 - high)) + high)
            else:
                lut[i] = int((i - low) * ((high - low) / (high - low)) + low)
        
        self.processed_image = cv2.LUT(self.current_image, lut)
        self.display_image(self.processed_image, self.processed_canvas)
    
    def apply_gamma(self):
        """Biến đổi Gamma (Power-Law)"""
        if not self.image_loaded:
            return
        
        c = self.gamma_c.get()
        gamma = self.gamma_val.get()

        img_normalized = self.current_image / 255.0

        gamma_corrected = c * np.power(img_normalized, gamma)
 
        gamma_corrected = np.clip(gamma_corrected * 255, 0, 255)
        self.processed_image = gamma_corrected.astype(np.uint8)
        
        self.display_image(self.processed_image, self.processed_canvas)
    
    def apply_smoothing(self):
        """Làm trơn ảnh bằng Average Filter (Low-pass)"""
        if not self.image_loaded:
            return
        
        kernel_size = int(self.smooth_size.get())
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.processed_image = cv2.blur(self.current_image, (kernel_size, kernel_size))
        self.display_image(self.processed_image, self.processed_canvas)
    
    def apply_gaussian(self):
        """Làm trơn ảnh bằng Gaussian Filter"""
        if not self.image_loaded:
            return
        
        kernel_size = int(self.gauss_size.get())
        sigma = self.gauss_sigma.get()
        
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.processed_image = cv2.GaussianBlur(
            self.current_image, 
            (kernel_size, kernel_size), 
            sigma
        )
        self.display_image(self.processed_image, self.processed_canvas)
    
    def apply_median(self):
        """Làm trơn ảnh bằng Median Filter"""
        if not self.image_loaded:
            return
        
        kernel_size = int(self.median_size.get())
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.processed_image = cv2.medianBlur(self.current_image, kernel_size)
        self.display_image(self.processed_image, self.processed_canvas)
    
    def apply_histogram_eq(self):
        """Cân bằng Histogram (CLAHE - Contrast Limited Adaptive Histogram Equalization)"""
        if not self.image_loaded:
            return
        
        clip_limit = self.hist_clip.get()
        
        ycrcb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2YCrCb)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        
        self.processed_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
        self.display_image(self.processed_image, self.processed_canvas)
    
    def update_display(self):
        """Cập nhật hiển thị - apply processed image lên current"""
        if self.processed_image is not None:
            self.current_image = self.processed_image.copy()
            messagebox.showinfo("Cập nhật", "Đã cập nhật ảnh xử lý làm ảnh gốc mới!")
        else:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh xử lý nào!")
    
    def reset_image(self):
        """Reset về ảnh gốc ban đầu"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.processed_canvas.delete("all")
            self.display_image(self.current_image, self.original_canvas)
            messagebox.showinfo("Reset", "Đã reset về ảnh gốc!")
        else:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh nào!")
    
    def save_image(self):
        """Lưu ảnh đã xử lý ra file"""
        if self.processed_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh xử lý nào để lưu!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), 
                      ("JPEG files", "*.jpg"),
                      ("All files", "*.*")]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{file_path}")


if __name__ == "__main__":
    root = Tk()
    app = ImageProcessingApp(root)
    root.mainloop()