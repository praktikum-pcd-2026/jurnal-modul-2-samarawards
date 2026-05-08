import numpy as np  
import matplotlib.pyplot as plt
import cv2


# manual grayscale (tanpa cv2.cvtColor grayscale)
gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        r, g, b = img[i, j]
        gray[i, j] = int(0.299*r + 0.587*g + 0.114*b)

plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')


# load background
bg = cv2.imread('langit.jpg')
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

# resize biar sama
bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]))

masking = np.zeros_like(img)

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        if gray[i, j] > 200:  # threshold bisa disesuaikan
            masking[i, j] = bg[i, j]
        else:
            masking[i, j] = img[i, j]

plt.imshow(masking)
plt.title("Masking Background")
plt.axis('off')


def spesifikasi_histogram(img_asli, img_target):
    # konversi ke grayscale di awal jika masih 3 channel
    if len(img_asli.shape) == 3:
        img_asli = cv2.cvtColor(img_asli.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if len(img_target.shape) == 3:
        img_target = cv2.cvtColor(img_target.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    hist_asli = np.zeros(256, dtype=int)
    hist_target = np.zeros(256, dtype=int)

    for i in range(img_asli.shape[0]):
        for j in range(img_asli.shape[1]):
            hist_asli[img_asli[i][j]] += 1

    for i in range(img_target.shape[0]):
        for j in range(img_target.shape[1]):
            hist_target[img_target[i][j]] += 1

    cdf_asli = np.zeros(256, dtype=float)
    cdf_target = np.zeros(256, dtype=float)
    cdf_asli[0] = hist_asli[0]
    cdf_target[0] = hist_target[0]

    for i in range(1, 256):
        cdf_asli[i] = cdf_asli[i-1] + hist_asli[i]
        cdf_target[i] = cdf_target[i-1] + hist_target[i]

    cdf_asli = cdf_asli / cdf_asli[-1]
    cdf_target = cdf_target / cdf_target[-1]

    map_hist = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        selisih = np.abs(cdf_target - cdf_asli[i])
        map_hist[i] = np.argmin(selisih)

    # for i in range(256):
    #     selisih_terkecil = float('inf')  # mulai dengan nilai tak terhingga
    #     j_terbaik = 0
    #     for j in range(256):
    #         selisih = abs(float(cdf_target[j]) - float(cdf_asal[i]))
    #         if selisih < selisih_terkecil:
    #             selisih_terkecil = selisih
    #             j_terbaik = j
    #     map_hist[i] = j_terbaik

    h, w = img_asli.shape[:2]
    hasil = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            hasil[i][j] = map_hist[img_asli[i][j]]

    return hasil

def spesifikasi_histogram(citra_asal, citra_target):

    # Histogram
    hist_asal = np.zeros(256, dtype=int)
    hist_target = np.zeros(256, dtype=int)

    # Hitung histogram citra asal
    for i in range(citra_asal.shape[0]):
        for j in range(citra_asal.shape[1]):
            pixel = citra_asal[i, j]
            hist_asal[pixel] += 1

    # Hitung histogram citra target
    for i in range(citra_target.shape[0]):
        for j in range(citra_target.shape[1]):
            pixel = citra_target[i, j]
            hist_target[pixel] += 1

    # CDF
    cdf_asal = np.zeros(256, dtype=float)
    cdf_target = np.zeros(256, dtype=float)

    cdf_asal[0] = hist_asal[0]
    cdf_target[0] = hist_target[0]

    # Hitung CDF
    for i in range(1, 256):
        cdf_asal[i] = cdf_asal[i - 1] + hist_asal[i]
        cdf_target[i] = cdf_target[i - 1] + hist_target[i]

    # Normalisasi CDF
    cdf_asal = cdf_asal / cdf_asal[-1]
    cdf_target = cdf_target / cdf_target[-1]

    # Mapping histogram
    map_hist = np.zeros(256, dtype=np.uint8)

    for i in range(256):

        selisih_min = abs(cdf_asal[i] - cdf_target[0])
        index = 0

        for j in range(1, 256):

            selisih = abs(cdf_asal[i] - cdf_target[j])

            if selisih < selisih_min:
                selisih_min = selisih
                index = j

        map_hist[i] = index

    # Terapkan mapping ke citra asal
    height, width = citra_asal.shape
    hasil = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):

            pixel = citra_asal[i, j]
            hasil[i, j] = map_hist[pixel]

    return hasil
#cara manggil: spesifikasi1 =
hasil_spek = spesifikasi_histogram(new, propaganda)


def ekualisasi(citra):

    height, width = citra.shape

    # Histogram
    hist = np.zeros(256, dtype=int)

    # Hitung histogram citra
    for i in range(height):
        for j in range(width):

            pixel = citra[i, j]
            hist[pixel] += 1

    # CDF
    cdf = np.zeros(256, dtype=int)

    cdf[0] = hist[0]

    # Hitung CDF
    for i in range(1, 256):

        cdf[i] = cdf[i - 1] + hist[i]

    # Normalisasi CDF
    cdf_normal = np.round(cdf * 255 / (height * width)).astype(np.uint8)

    # Hasil ekualisasi
    hasil = np.zeros_like(citra, dtype=np.uint8)

    # Terapkan hasil CDF normalisasi
    for i in range(height):
        for j in range(width):

            pixel = citra[i, j]
            hasil[i, j] = cdf_normal[pixel]

    return hasil

hasil_eku = ekualisasi(new)