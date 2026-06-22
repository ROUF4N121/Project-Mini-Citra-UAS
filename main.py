import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import glob

class AplikasiPengolahanCitra:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Pengolahan Citra - UAS")
        self.root.geometry("500x600")
        self.root.configure(bg="#f0f0f0")
        
        self.image_path = None
        self.folder_path = None

        # Judul Aplikasi
        tk.Label(root, text="Aplikasi Pengolahan Citra Daun", 
                 font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=15)

        # Panel untuk menampilkan gambar yang dipilih
        self.panel_gambar = tk.Label(root, bg="#e0e0e0", text="Belum ada data dipilih", 
                                     width=45, height=15, relief="groove")
        self.panel_gambar.pack(pady=10)

        # Frame untuk Mode 1 (Satu Gambar)
        frame_single = tk.LabelFrame(root, text="Mode 1: Analisis 1 Gambar", bg="#f0f0f0")
        frame_single.pack(pady=10, padx=20, fill="x")
        
        tk.Button(frame_single, text="Pilih 1 Gambar", command=self.pilih_gambar, 
                  width=15, bg="#4CAF50", fg="white").pack(side="left", padx=10, pady=10)
        tk.Button(frame_single, text="Tampilkan Hasil", command=self.proses_satu_gambar, 
                  width=15, bg="#2196F3", fg="white").pack(side="right", padx=10, pady=10)

        # Frame untuk Mode 2 (Dataset 20 Gambar)
        frame_batch = tk.LabelFrame(root, text="Mode 2: Proses Satu Folder", bg="#f0f0f0")
        frame_batch.pack(pady=10, padx=20, fill="x")

        tk.Button(frame_batch, text="Pilih Folder Dataset", command=self.pilih_folder, 
                  width=15, bg="#FF9800", fg="white").pack(side="left", padx=10, pady=10)
        tk.Button(frame_batch, text="Proses & Simpan Semua", command=self.proses_semua_gambar, 
                  width=20, bg="#E91E63", fg="white").pack(side="right", padx=10, pady=10)

    # --- FUNGSI MODE 1 ---
    def pilih_gambar(self):
        self.image_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("File Gambar", "*.jpg *.jpeg *.png")]
        )
        if self.image_path:
            self.folder_path = None # Reset folder path
            img = Image.open(self.image_path)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.panel_gambar.configure(image=img_tk, text="")
            self.panel_gambar.image = img_tk

    def proses_satu_gambar(self):
        if not self.image_path:
            messagebox.showwarning("Peringatan", "Pilih 1 gambar terlebih dahulu!")
            return
        self._jalankan_pengolahan(self.image_path, show_plot=True)

    # --- FUNGSI MODE 2 ---
    def pilih_folder(self):
        self.folder_path = filedialog.askdirectory(title="Pilih Folder Dataset")
        if self.folder_path:
            self.image_path = None # Reset image path
            self.panel_gambar.configure(image='', text=f"Folder Terpilih:\n{self.folder_path}\n\nSiap memproses semua gambar di dalamnya.")

    def proses_semua_gambar(self):
        if not self.folder_path:
            messagebox.showwarning("Peringatan", "Pilih folder dataset terlebih dahulu!")
            return
            
        # Mencari semua file jpg dan png di dalam folder
        file_gambar = glob.glob(os.path.join(self.folder_path, "*.jpg")) + \
                      glob.glob(os.path.join(self.folder_path, "*.jpeg")) + \
                      glob.glob(os.path.join(self.folder_path, "*.png"))
                      
        if len(file_gambar) == 0:
            messagebox.showerror("Error", "Tidak ditemukan file gambar di folder tersebut.")
            return

        # Membuat folder baru untuk menyimpan hasil
        folder_hasil = os.path.join(self.folder_path, "Hasil_Pengolahan")
        os.makedirs(folder_hasil, exist_ok=True)

        messagebox.showinfo("Proses Dimulai", f"Memproses {len(file_gambar)} gambar...\nMohon tunggu, aplikasi mungkin terlihat 'Not Responding' selama beberapa detik.")

        # Looping ke seluruh 20+ gambar
        for path in file_gambar:
            nama_file = os.path.basename(path)
            path_simpan = os.path.join(folder_hasil, f"hasil_{nama_file}")
            self._jalankan_pengolahan(path, show_plot=False, save_path=path_simpan)
            
        messagebox.showinfo("Selesai!", f"Berhasil memproses {len(file_gambar)} gambar!\nHasilnya sudah disimpan di folder:\n{folder_hasil}")

    # --- INTI PENGOLAHAN CITRA ---
    def _jalankan_pengolahan(self, path, show_plot=True, save_path=None):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1-5. Proses Citra
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, biner = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        pixel_values = np.float32(img_rgb.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        segmented_image = np.uint8(centers)[labels.flatten()].reshape((img_rgb.shape))

        # Visualisasi Matplotlib
        titles = ['Asli', 'Grayscale', 'Biner', 'Hist. Equalization', 'Gaussian', 'Canny', 'K-Means']
        images = [img_rgb, gray, biner, equalized, blurred, edges, segmented_image]
        
        plt.figure(figsize=(15, 8))
        plt.suptitle(f"Hasil Pengolahan: {os.path.basename(path)}", fontsize=16, fontweight='bold')
        for i in range(7):
            plt.subplot(2, 4, i+1)
            if len(images[i].shape) == 2:
                plt.imshow(images[i], cmap='gray')
            else:
                plt.imshow(images[i])
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
            
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        elif save_path:
            plt.savefig(save_path) # Simpan grafik menjadi file JPG/PNG baru
            plt.close() # Tutup memori agar tidak memberatkan komputer

if __name__ == "__main__":
    root = tk.Tk()
    app = AplikasiPengolahanCitra(root)
    root.mainloop()