import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter, ImageOps, ImageEnhance
import cv2
import numpy as np

# Hàm để tải ảnh lên
def load_image(label):
    filepath = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.jpeg;*.png;*.bmp")])
    if filepath:
        image = Image.open(filepath)
        # Lưu ảnh gốc để sử dụng cho việc lưu sau này
        label.original_image = image.copy()  # Lưu ảnh gốc
        # Giới hạn kích thước ảnh khi hiển thị trong giao diện
        image.thumbnail((800, 800))  # Giới hạn kích thước ảnh hiển thị
        photo = ImageTk.PhotoImage(image)
        label.image = photo
        label.configure(image=photo, text="")  # Xóa chữ khi ảnh được tải
        label.filepath = filepath  # Lưu đường dẫn ảnh để xử lý sau

# Hàm xử lý ảnh
def process_image():
    if hasattr(original_image_label, "original_image"):
        technique = techniques_combobox.get()
        image = original_image_label.original_image.copy()

        # Chuyển ảnh PIL sang định dạng NumPy để xử lý với OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        try:
            kernel_size = int(kernel_size_entry.get())
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("Kernel size phải là số lẻ và lớn hơn 0.")
        except ValueError as e:
            status_label.configure(text=f"Lỗi: {e}")
            return

        # Bộ lọc ảnh
        if technique == "Blur":
            intensity = int(filter_slider.get())
            processed_image = image.filter(ImageFilter.GaussianBlur(radius=intensity))
        elif technique == "Sharpen":
            intensity = int(filter_slider.get())
            enhancer = ImageEnhance.Sharpness(image)
            processed_image = enhancer.enhance(1 + intensity * 0.5)
        elif technique == "Grayscale":
            intensity = int(filter_slider.get())
            processed_image = ImageOps.grayscale(image).convert("RGB")
            processed_image = ImageEnhance.Brightness(processed_image).enhance(1 + intensity * 0.1)
        elif technique == "Invert Colors":
            intensity = int(filter_slider.get()) / 10  # Tỷ lệ hòa trộn
            inverted = cv2.bitwise_not(image_cv)  # Đảo ngược màu hoàn toàn
            blended = cv2.addWeighted(image_cv, 1 - intensity, inverted, intensity, 0)  # Hòa trộn
            processed_image = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        
        elif technique == "Histogram Equalization":
            intensity = int(filter_slider.get())
            image_yuv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2YUV)
            image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
            equalized = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
            processed_image = Image.fromarray(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
            # Điều chỉnh độ sáng sau cân bằng Histogram
            enhancer = ImageEnhance.Brightness(processed_image)
            processed_image = enhancer.enhance(1 + intensity * 0.1)
        elif technique == "Interpolation":
            scale = 1 + filter_slider.get() / 5  # Tỷ lệ phóng to/thu nhỏ
            height, width = image_cv.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Thay đổi kích thước ảnh bằng nội suy
            resized = cv2.resize(image_cv, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            processed_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            # Hiển thị ảnh phóng to/thu nhỏ nhưng không giới hạn lại với thumbnail
            photo = ImageTk.PhotoImage(processed_image)
            processed_image_label.image = photo
            processed_image_label.configure(image=photo, text="")
        elif technique == "Mean Filter":
            mean_filtered = cv2.blur(image_cv, (kernel_size, kernel_size))
            processed_image = Image.fromarray(cv2.cvtColor(mean_filtered, cv2.COLOR_BGR2RGB))
        elif technique == "Gaussian Filter":
            gaussian_filtered = cv2.GaussianBlur(image_cv, (kernel_size, kernel_size), 0)
            processed_image = Image.fromarray(cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB))
        elif technique == "Median Filter":
            median_filtered = cv2.medianBlur(image_cv, kernel_size)
            processed_image = Image.fromarray(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))

        # Phát hiện biên
        elif technique == "Laplacian":
            intensity = int(filter_slider.get())
            laplacian = cv2.Laplacian(image_cv, cv2.CV_64F, ksize=2 * intensity + 1)
            laplacian = cv2.convertScaleAbs(laplacian)
            processed_image = Image.fromarray(cv2.cvtColor(laplacian, cv2.COLOR_BGR2RGB))
        elif technique == "Gradient (Sobel)":
            intensity = int(filter_slider.get())
            sobel_x = cv2.Sobel(image_cv, cv2.CV_64F, 1, 0, ksize=2 * intensity + 1)
            sobel_y = cv2.Sobel(image_cv, cv2.CV_64F, 0, 1, ksize=2 * intensity + 1)
            sobel = cv2.magnitude(sobel_x, sobel_y)
            sobel = cv2.convertScaleAbs(sobel)
            processed_image = Image.fromarray(cv2.cvtColor(sobel, cv2.COLOR_BGR2RGB))
            processed_image = Image.fromarray(cv2.cvtColor(sobel, cv2.COLOR_BGR2RGB))
        elif technique == "Canny":
            threshold1 = int(filter_slider.get()) * 10
            threshold2 = threshold1 * 2
            canny_edges = cv2.Canny(image_cv, threshold1, threshold2)
            processed_image = Image.fromarray(cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB))

        # Phân ngưỡng
        elif technique == "Otsu Threshold":
            intensity = int(filter_slider.get())
            gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            _, otsu_thresh = cv2.threshold(
                gray_image, intensity * 5, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            processed_image = Image.fromarray(cv2.cvtColor(otsu_thresh, cv2.COLOR_GRAY2RGB))

        elif technique == "Fixed Threshold":
            gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
            threshold_value = int(filter_slider.get()) * 25  # Giá trị ngưỡng tùy chỉnh
            _, fixed_thresh = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            processed_image = Image.fromarray(cv2.cvtColor(fixed_thresh, cv2.COLOR_GRAY2RGB))

        elif technique == "Adaptive Threshold":
            gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
            block_size = max(3, int(filter_slider.get()) * 2 + 1)  # Kích thước khối (luôn lẻ)
            adaptive_thresh = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 5
            )
            processed_image = Image.fromarray(cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2RGB))

        # Xoay ảnh
        elif technique == "Rotate":
            angle = int(rotation_slider.get())
            processed_image = image.rotate(-angle, expand=True)
        elif technique == "Flip Vertical":
            processed_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif technique == "Flip Horizontal":
            processed_image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Di chuyển ảnh
        elif technique == "Translate":
            x = int(x_entry.get())
            y = int(y_entry.get())
            new_image = Image.new("RGB", (image.width + abs(x), image.height + abs(y)), (255, 255, 255))
            new_image.paste(image, (max(0, x), max(0, y)))
            processed_image = new_image

        # Chinh sua mau sac
        elif technique == "Color Adjustment":
            hue = int(hue_slider.get())  # Giá trị Hue
            saturation = int(saturation_slider.get()/100)   # Giá trị Saturation (0.0 - 2.0)
            brightness = int(brightness_slider.get()/100)   # Giá trị Brightness (0.0 - 2.0)

            # Chuyển đổi ảnh sang không gian HSV
            hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV).astype("float32")

            # Điều chỉnh Hue
            hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue) % 180

            # Điều chỉnh Saturation
            hsv_image[:, :, 1] *= saturation
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)

            # Điều chỉnh Brightness
            hsv_image[:, :, 2] *= brightness
            hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)

            # Chuyển về không gian màu BGR
            adjusted_image = cv2.cvtColor(hsv_image.astype("uint8"), cv2.COLOR_HSV2BGR)

            # Chuyển thành Image để hiển thị
            processed_image = Image.fromarray(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))    

        else:
            processed_image = image

        # Hiển thị ảnh sau xử lý (vẫn giữ kích thước nhỏ hơn)
        processed_image.thumbnail((800,800))
        photo = ImageTk.PhotoImage(processed_image)
        processed_image_label.image = photo
        processed_image_label.configure(image=photo, text="")
        status_label.configure(text="Xử lý ảnh thành công!")

        # Lưu ảnh đã xử lý vào thuộc tính để có thể lưu sau
        processed_image_label.processed_image = processed_image



# Hàm cập nhật giá trị góc xoay
def update_rotation_label(value):
    rotation_value_label.configure(text=f"Angle: {int(float(value))}°")

# Hàm lưu ảnh
def save_image():
    if hasattr(processed_image_label, "processed_image"):
        # Lưu ảnh đã xử lý với độ phân giải gốc
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            processed_image_label.processed_image.save(file_path)  # Lưu ảnh đã xử lý với kích thước gốc
            status_label.configure(text="Ảnh đã được lưu thành công!")

# Tạo cửa sổ chính
app = ctk.CTk()
app.title("Image Processing App")
app.geometry("1920x1080")  # Kích thước cửa sổ 

# Tạo một frame cuộn
scrollable_frame = ctk.CTkScrollableFrame(app, width=1600, height=800)
scrollable_frame.grid(row=0, column=0, columnspan=2)

# Giao diện 2 ô ảnh
original_image_label = ctk.CTkLabel(scrollable_frame, text="Original Image", width=650, height=700, corner_radius=10)
original_image_label.grid(row=0, column=0, padx=30, pady=30)

processed_image_label = ctk.CTkLabel(scrollable_frame, text="Processed Image", width=650, height=700, corner_radius=10)
processed_image_label.grid(row=0, column=1, padx=30, pady=30)

# Nút tải ảnh lên
upload_button = ctk.CTkButton(scrollable_frame, text="Upload Image", width=20, height=40, command=lambda: load_image(original_image_label))
upload_button.grid(row=1, column=0, padx=30, pady=20, sticky="ew")

# Nút xử lý ảnh
process_button = ctk.CTkButton(scrollable_frame, text="Process Image", width=20, height=40, command=process_image)
process_button.grid(row=1, column=1, padx=30, pady=20, sticky="ew")

# Thanh kéo thả tham số filter intensity
filter_slider = ctk.CTkSlider(scrollable_frame, from_=0, to=10, width=300)
filter_slider.set(2)  # Giá trị mặc định
filter_slider_label = ctk.CTkLabel(scrollable_frame, text="Intensity (Cường độ):")
filter_slider_label.grid(row=3, column=0, sticky="e", pady=10)
filter_slider.grid(row=3, column=1, sticky="w")

# Thanh kéo thả góc xoay
rotation_slider = ctk.CTkSlider(scrollable_frame, from_=0, to=360, width=300, command=update_rotation_label)
rotation_slider.set(0)
rotation_slider_label = ctk.CTkLabel(scrollable_frame, text="Rotation Angle:")
rotation_slider_label.grid(row=4, column=0, sticky="e", pady=10)
rotation_slider.grid(row=4, column=1, sticky="w")

# Hiển thị giá trị góc xoay
rotation_value_label = ctk.CTkLabel(scrollable_frame, text="Angle: 0°", font=("Arial", 12))
rotation_value_label.grid(row=5, column=1, sticky="w", pady=5)

# Khu vực nhập tọa độ di chuyển
x_label = ctk.CTkLabel(scrollable_frame, text="X Offset:")
x_label.grid(row=12, column=0, sticky="e", pady=5)
x_entry = ctk.CTkEntry(scrollable_frame, width=200)
x_entry.grid(row=12, column=1, pady=5, sticky="w")

y_label = ctk.CTkLabel(scrollable_frame, text="Y Offset:")
y_label.grid(row=13, column=0, sticky="e", pady=5)
y_entry = ctk.CTkEntry(scrollable_frame, width=200)
y_entry.grid(row=13, column=1, pady=5, sticky="w")

# Ô nhập kích thước kernel
kernel_size_label = ctk.CTkLabel(scrollable_frame, text="Kernel Size:")
kernel_size_label.grid(row=14, column=0, sticky="e", pady=5)

kernel_size_entry = ctk.CTkEntry(scrollable_frame, width=200)
kernel_size_entry.grid(row=14, column=1, pady=5, sticky="w")
kernel_size_entry.insert(0, "3")  # Giá trị mặc định


# Bổ sung các thanh trượt cho Hue, Saturation và Brightness
# Thêm hàm cập nhật giá trị
def update_hue_value(value):
    hue_value_label.configure(text=f"Hue: {int(float(value))}")

def update_brightness_value(value):
    brightness_value_label.configure(text=f"Brightness: {int(float(value))}")

def update_saturation_value(value):
    saturation_value_label.configure(text=f"Saturation: {int(float(value))}")

# Thêm thanh Hue
hue_slider = ctk.CTkSlider(scrollable_frame, from_=-90, to=90, width=300, command=update_hue_value)
hue_slider.set(0)
hue_slider_label = ctk.CTkLabel(scrollable_frame, text="Hue (tông màu)):")
hue_slider_label.grid(row=6, column=0, sticky="e", pady=10)
hue_slider.grid(row=6, column=1, sticky="w")
hue_value_label = ctk.CTkLabel(scrollable_frame, text="Hue: 0", font=("Arial", 12))
hue_value_label.grid(row=7, column=1, sticky="w", padx=10)

# Thêm thanh Brightness
brightness_slider = ctk.CTkSlider(scrollable_frame, from_=0, to=200, width=300, command=update_brightness_value)
brightness_slider.set(100)
brightness_slider_label = ctk.CTkLabel(scrollable_frame, text="Brightness (độ sáng)):")
brightness_slider_label.grid(row=8, column=0, sticky="e", pady=10)
brightness_slider.grid(row=8, column=1, sticky="w")
brightness_value_label = ctk.CTkLabel(scrollable_frame, text="Brightness: 0", font=("Arial", 12))
brightness_value_label.grid(row=9, column=1, sticky="w", padx=10)

# Thêm thanh Saturation
saturation_slider = ctk.CTkSlider(scrollable_frame, from_=0, to=200, width=300, command=update_saturation_value)
saturation_slider.set(100)
saturation_slider_label = ctk.CTkLabel(scrollable_frame, text="Saturation (Độ bão hòa)):")
saturation_slider_label.grid(row=10, column=0, sticky="e", pady=10)
saturation_slider.grid(row=10, column=1, sticky="w")
saturation_value_label = ctk.CTkLabel(scrollable_frame, text="Saturation: 100", font=("Arial", 12))
saturation_value_label.grid(row=11, column=1, sticky="w", padx=10)




# ComboBox chọn thuật toán
techniques_label = ctk.CTkLabel(scrollable_frame, text="Select Processing Technique:")
techniques_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
techniques_combobox = ctk.CTkComboBox(scrollable_frame, values=[
    "Blur",
    "Sharpen",
    "Grayscale",
    "Invert Colors",
    "Histogram Equalization",
    "Interpolation",
    "Mean Filter",
    "Gaussian Filter",
    "Median Filter",
    "Laplacian",
    "Gradient (Sobel)",
    "Canny", 
    "Otsu Threshold",
    "Fixed Threshold",
    "Adaptive Threshold",
    "Rotate",
    "Flip Vertical",
    "Flip Horizontal",
    "Translate",
   "Color Adjustment" 
])




techniques_combobox.grid(row=2, column=1, pady=20)


# Label trạng thái
status_label = ctk.CTkLabel(scrollable_frame, text="", font=("Arial", 12), text_color="red")
status_label.grid(row=15, column=0, columnspan=2, pady=10)

# Nút lưu ảnh (đưa xuống cuối cùng)
save_button = ctk.CTkButton(scrollable_frame, text="Save Image", width=20, height=40, command=save_image)
save_button.grid(row=16, column=0, columnspan=2, pady=30, sticky="ew")

# Cấu hình cột
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=2)

# Chạy ứng dụng
app.mainloop()