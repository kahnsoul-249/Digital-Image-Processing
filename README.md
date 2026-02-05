# Digital-Image-Processing
Homework and all knowledge about this subject
# DIGITAL IMAGE PROCESSING - TÀI LIỆU HƯỚNG DẪN

## 📚 MỤC LỤC
1. [Tổng quan ứng dụng](#tổng-quan)
2. [Cài đặt và chạy](#cài-đặt)
3. [Chi tiết các phép xử lý ảnh](#chi-tiết-xử-lý)
4. [Công thức toán học](#công-thức)
5. [Hướng dẫn sử dụng](#hướng-dẫn)

---

## 📖 TỔNG QUAN ỨNG DỤNG

Ứng dụng GUI xử lý ảnh số với đầy đủ các chức năng:

### **1. Image Transformations (Biến đổi cường độ)**
- Negative Image (Ảnh âm bản)
- Log Transformation (Biến đổi Logarit)
- Piecewise-Linear (Contrast Stretching)
- Gamma Correction (Hiệu chỉnh Gamma)

### **2. Spatial Filtering (Lọc không gian)**
- **Low-pass filters** (Lọc thông thấp - làm mịn):
  - Average Filter (Lọc trung bình)
  - Gaussian Filter (Lọc Gauss)
  - Median Filter (Lọc trung vị)

### **3. Histogram Processing**
- CLAHE - Contrast Limited Adaptive Histogram Equalization
- Cân bằng độ sáng cục bộ

---

## 🚀 CÀI ĐẶT VÀ CHẠY

### **Bước 1: Cài đặt thư viện**
```bash
pip install opencv-python numpy pillow
```

### **Bước 2: Chạy ứng dụng**
```bash
python digital_image_processing_gui.py
```

### **Bước 3: Sử dụng**
1. Click "Chọn ảnh" để load ảnh
2. Điều chỉnh các slider để xem hiệu ứng real-time
3. Click "Cập nhật" để apply kết quả lên ảnh gốc
4. Click "Lưu ra file" để save ảnh

---

## 🔬 CHI TIẾT CÁC PHÉP XỬ LÝ ẢNH

### **1. NEGATIVE IMAGE (ẢNH ÂM BẢN)**

**Mục đích:** Đảo ngược cường độ sáng của ảnh

**Công thức:**
```
s = 255 - r
```
Trong đó:
- `r`: giá trị pixel gốc (0-255)
- `s`: giá trị pixel sau khi xử lý

**Ứng dụng:**
- Tăng cường chi tiết vùng tối
- Y học: đọc ảnh X-quang
- Phân tích ảnh grayscale

**Code implementation:**
```python
def apply_negative(self):
    self.processed_image = 255 - self.current_image
```

---

### **2. LOG TRANSFORMATION (BIẾN ĐỔI LOGARIT)**

**Mục đích:** Mở rộng giá trị pixel tối, nén giá trị pixel sáng

**Công thức:**
```
s = c × log(1 + r)
```
Trong đó:
- `c`: hằng số scaling (thường = 255 / log(256))
- `r`: giá trị pixel gốc
- `s`: giá trị pixel sau xử lý

**Đặc điểm:**
- Tăng cường vùng tối
- Giảm contrast vùng sáng
- Curve lõm (concave)

**Ứng dụng:**
- Hiển thị ảnh Fourier spectrum
- Ảnh có dynamic range rộng
- Ảnh chụp trong điều kiện thiếu sáng

**Code implementation:**
```python
def apply_log(self):
    c = self.log_c.get()
    img_float = self.current_image.astype(np.float32)
    log_image = c * np.log1p(img_float)  # log1p = log(1 + x)
    self.processed_image = np.clip(log_image, 0, 255).astype(np.uint8)
```

---

### **3. PIECEWISE-LINEAR TRANSFORMATION (CONTRAST STRETCHING)**

**Mục đích:** Tăng contrast bằng cách kéo giãn histogram

**Công thức:**
```
Chia làm 3 đoạn:
- r < r1: s = (s1/r1) × r
- r1 ≤ r ≤ r2: s = ((s2-s1)/(r2-r1)) × (r-r1) + s1
- r > r2: s = ((255-s2)/(255-r2)) × (r-r2) + s2
```

**Tham số:**
- `(r1, s1)`: điểm thấp
- `(r2, s2)`: điểm cao

**Đặc điểm:**
- Tăng contrast vùng quan tâm
- Nén contrast vùng không quan tâm

**Ứng dụng:**
- Tăng cường ảnh có contrast thấp
- Satellite imaging
- Medical imaging

**Code implementation:**
```python
def apply_piecewise(self):
    low = self.piece_low.get()
    high = self.piece_high.get()
    
    # Tạo lookup table
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < low:
            lut[i] = int(i * (low / 256))
        elif i > high:
            lut[i] = int((i - high) * ((255 - high) / (255 - high)) + high)
        else:
            lut[i] = int((i - low) * ((high - low) / (high - low)) + low)
    
    self.processed_image = cv2.LUT(self.current_image, lut)
```

---

### **4. GAMMA CORRECTION (HIỆU CHỈNH GAMMA)**

**Mục đích:** Điều chỉnh độ sáng phi tuyến

**Công thức:**
```
s = c × r^γ
```
Trong đó:
- `c`: hằng số (thường = 1)
- `γ` (gamma):
  - γ < 1: làm sáng ảnh (curve lồi)
  - γ = 1: không thay đổi
  - γ > 1: làm tối ảnh (curve lõm)

**Ứng dụng:**
- Hiệu chỉnh gamma màn hình
- Tương thích với human perception
- Tiền xử lý cho machine learning

**Code implementation:**
```python
def apply_gamma(self):
    c = self.gamma_c.get()
    gamma = self.gamma_val.get()
    
    # Chuẩn hóa về [0, 1]
    img_normalized = self.current_image / 255.0
    
    # Áp dụng gamma: s = c * r^gamma
    gamma_corrected = c * np.power(img_normalized, gamma)
    
    # Scale về [0, 255]
    self.processed_image = np.clip(gamma_corrected * 255, 0, 255).astype(np.uint8)
```

---

### **5. AVERAGE FILTER (LỌC TRUNG BÌNH - LOW-PASS)**

**Mục đích:** Làm mịn ảnh bằng cách lấy trung bình các pixel lân cận

**Công thức:**
```
g(x,y) = (1/MN) × Σ f(s,t)
```
Trong đó:
- `M×N`: kích thước kernel
- `f(s,t)`: giá trị pixel trong vùng lân cận

**Kernel mẫu 3×3:**
```
1/9 [1 1 1]
    [1 1 1]
    [1 1 1]
```

**Đặc điểm:**
- Giảm noise
- Làm mờ ảnh
- Simple và nhanh

**Ứng dụng:**
- Khử nhiễu cơ bản
- Tiền xử lý ảnh

**Code implementation:**
```python
def apply_smoothing(self):
    kernel_size = int(self.smooth_size.get())
    if kernel_size % 2 == 0:
        kernel_size += 1  # Đảm bảo lẻ
    
    self.processed_image = cv2.blur(self.current_image, 
                                    (kernel_size, kernel_size))
```

---

### **6. GAUSSIAN FILTER (LỌC GAUSS - LOW-PASS)**

**Mục đích:** Làm mịn ảnh với trọng số theo phân phối Gauss

**Công thức:**
```
G(x,y) = (1/2πσ²) × e^(-(x²+y²)/(2σ²))
```
Trong đó:
- `σ` (sigma): độ lệch chuẩn, điều khiển độ mịn
- Pixel gần center có trọng số cao hơn

**Kernel mẫu 3×3 (σ≈1):**
```
1/16 [1 2 1]
     [2 4 2]
     [1 2 1]
```

**Đặc điểm:**
- Smooth tự nhiên hơn average filter
- Giữ edge tốt hơn
- Gaussian noise reduction

**Ứng dụng:**
- Image preprocessing cho edge detection
- Noise reduction
- Image pyramids

**Code implementation:**
```python
def apply_gaussian(self):
    kernel_size = int(self.gauss_size.get())
    sigma = self.gauss_sigma.get()
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    self.processed_image = cv2.GaussianBlur(
        self.current_image,
        (kernel_size, kernel_size),
        sigma
    )
```

---

### **7. MEDIAN FILTER (LỌC TRUNG VỊ - NON-LINEAR)**

**Mục đích:** Khử nhiễu "salt and pepper" bằng cách lấy median

**Công thức:**
```
g(x,y) = median{f(s,t)}
```
Trong đó:
- Sắp xếp các pixel trong window
- Chọn giá trị ở giữa (median)

**Đặc điểm:**
- **NON-LINEAR** filter
- Rất hiệu quả với impulse noise
- Bảo toàn edge tốt

**Ứng dụng:**
- Khử salt-and-pepper noise
- Medical image processing
- Preprocessing cho OCR

**Code implementation:**
```python
def apply_median(self):
    kernel_size = int(self.median_size.get())
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    self.processed_image = cv2.medianBlur(self.current_image, 
                                          kernel_size)
```

---

### **8. HISTOGRAM EQUALIZATION (CÂN BẰNG HISTOGRAM)**

**Mục đích:** Phân bố lại histogram để tăng contrast toàn cục

**Phương pháp: CLAHE (Contrast Limited Adaptive Histogram Equalization)**

**Công thức cơ bản:**
```
s = T(r) = (L-1) × Σ(k=0 to r) P(k)
```
Trong đó:
- `P(k)`: xác suất của mức xám k
- `L`: số mức xám (256)
- Cumulative Distribution Function (CDF)

**CLAHE cải tiến:**
- Chia ảnh thành tiles (8×8)
- Equalize từng tile
- Clip histogram để tránh over-amplification
- Interpolate bilinear giữa các tiles

**Tham số:**
- `clipLimit`: giới hạn contrast (thường 2.0-4.0)
- `tileGridSize`: kích thước tile (thường 8×8)

**Đặc điểm:**
- Tăng contrast cục bộ
- Tránh over-enhancement
- Adaptive

**Ứng dụng:**
- Medical imaging (X-ray, CT, MRI)
- Underwater images
- Low-light photography

**Code implementation:**
```python
def apply_histogram_eq(self):
    clip_limit = self.hist_clip.get()
    
    # Chuyển sang YCrCb color space
    ycrcb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2YCrCb)
    
    # CLAHE chỉ áp dụng lên kênh Y (luminance)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                            tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    
    # Chuyển về BGR
    self.processed_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
```

---

## 📊 SO SÁNH CÁC BỘ LỌC

| Bộ lọc | Loại | Khử noise | Giữ edge | Tốc độ | Ứng dụng chính |
|--------|------|-----------|----------|--------|----------------|
| Average | Linear | Trung bình | Kém | Nhanh | Noise reduction cơ bản |
| Gaussian | Linear | Tốt | Khá tốt | Trung bình | Tiền xử lý, image pyramids |
| Median | Non-linear | Rất tốt (impulse) | Tốt nhất | Chậm | Salt-pepper noise |

---

## 🎯 HƯỚNG DẪN SỬ DỤNG

### **Workflow chuẩn:**

1. **Load ảnh:** Click "Chọn ảnh"
2. **Thử nghiệm:** Điều chỉnh các slider
3. **Apply:** Click nút hoặc slider tự động update
4. **Chain processing:**
   - Apply filter 1
   - Click "Cập nhật"
   - Apply filter 2
   - Click "Cập nhật"
   - ...
5. **Save:** Click "Lưu ra file"

### **Tips:**

- **Negative:** Dùng cho ảnh X-ray, phim âm bản
- **Log:** Tăng cường vùng tối trong ảnh có dynamic range lớn
- **Gamma:** 
  - γ < 1: làm sáng (0.3-0.8)
  - γ > 1: làm tối (1.2-3.0)
- **Piecewise:** Kéo giãn contrast vùng quan tâm
- **Smoothing:** Kernel lớn = mịn hơn nhưng mất chi tiết
- **Gaussian:** Sigma lớn = blur nhiều hơn
- **Median:** Tốt nhất cho salt-pepper noise
- **Histogram:** clipLimit cao = contrast mạnh hơn

---

## 🔧 CUSTOMIZATION

### **Thêm High-pass filter:**

```python
def apply_highpass(self):
    """High-pass filter = Original - Low-pass"""
    # Làm mịn bằng Gaussian
    blurred = cv2.GaussianBlur(self.current_image, (15, 15), 3)
    
    # High-pass = Original - Blurred
    self.processed_image = cv2.addWeighted(
        self.current_image, 2,  # Original × 2
        blurred, -1,             # - Blurred
        0
    )
    self.processed_image = np.clip(self.processed_image, 0, 255)
```

### **Thêm Laplacian filter:**

```python
def apply_laplacian(self):
    """Laplacian edge detection"""
    gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    self.processed_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
```

---

## 📚 TÀI LIỆU THAM KHẢO

1. **Digital Image Processing (Gonzalez & Woods)** - Bible của xử lý ảnh
2. **OpenCV Documentation** - https://docs.opencv.org/
3. **Numpy Documentation** - https://numpy.org/doc/

---

## ⚠️ LƯU Ý KỸ THUẬT

1. **Overflow handling:** Dùng `np.clip()` để đảm bảo giá trị trong [0, 255]
2. **Data type:** Chuyển đổi giữa `uint8`, `float32` cho tính toán chính xác
3. **Color space:** 
   - OpenCV dùng BGR (không phải RGB)
   - Histogram equalization tốt nhất trên YCrCb color space
4. **Kernel size:** Luôn là số lẻ (3, 5, 7, 9, ...)
5. **Performance:** Các phép toán vectorized (numpy) nhanh hơn loops

---

**Chúc bạn thành công với bài tập Digital Image Processing! 🎓**
