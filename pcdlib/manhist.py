import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def equalization(image):
    # histogram
    hist = np.zeros(256)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            hist[image[y, x]] += 1

    # normalisasi histogram
    hist = hist / image.size

    # cumulative distribution function
    cdf = np.zeros(256)
    cdf[0] = hist[0]

    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # mapping
    transform = np.round(cdf * 255).astype(np.uint8)

    # hasil equalization
    hasil = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            hasil[y, x] = transform[image[y, x]]

    return hasil

def ekualisasi(citra):

    height, width = citra.shape

    hist = np.zeros(256, dtype=int)

    # hitung histogram
    for y in range(height):
        for x in range(width):
            pixel = citra[y, x]
            hist[pixel] += 1

    cdf = np.zeros(256, dtype=int)
    cdf[0] = hist[0]

    # hitung CDF
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf_normal = np.round(
        cdf * 255 / (height * width)
    ).astype(np.uint8)

    hasil = np.zeros_like(citra, dtype=np.uint8)

    # mapping hasil ekualisasi
    for y in range(height):
        for x in range(width):
            hasil[y, x] = cdf_normal[citra[y, x]]

    return hasil

def specification(source, reference):

    # histogram source
    hist_source = np.zeros(256)

    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            hist_source[source[y, x]] += 1

    hist_source /= source.size

    # histogram reference
    hist_ref = np.zeros(256)

    for y in range(reference.shape[0]):
        for x in range(reference.shape[1]):
            hist_ref[reference[y, x]] += 1

    hist_ref /= reference.size

    # CDF source
    cdf_source = np.zeros(256)
    cdf_source[0] = hist_source[0]

    for i in range(1, 256):
        cdf_source[i] = cdf_source[i - 1] + hist_source[i]

    # CDF reference
    cdf_ref = np.zeros(256)
    cdf_ref[0] = hist_ref[0]

    for i in range(1, 256):
        cdf_ref[i] = cdf_ref[i - 1] + hist_ref[i]

    # mapping
    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):

        selisih = np.abs(cdf_source[i] - cdf_ref)
        mapping[i] = np.argmin(selisih)

    # hasil
    hasil = np.zeros_like(source)

    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            hasil[y, x] = mapping[source[y, x]]

    return hasil

def spesifikasi_histogram(citra_asal, citra_target):

    hist_asal = np.zeros(256, dtype=int)
    hist_target = np.zeros(256, dtype=int)

    # hitung histogram citra asal
    for y in range(citra_asal.shape[0]):
        for x in range(citra_asal.shape[1]):
            pixel = citra_asal[y, x]
            hist_asal[pixel] += 1

    # hitung histogram citra target
    for y in range(citra_target.shape[0]):
        for x in range(citra_target.shape[1]):
            pixel = citra_target[y, x]
            hist_target[pixel] += 1

    cdf_asal = np.zeros(256, dtype=float)
    cdf_target = np.zeros(256, dtype=float)

    cdf_asal[0] = hist_asal[0]
    cdf_target[0] = hist_target[0]

    # hitung CDF
    for i in range(1, 256):
        cdf_asal[i] = cdf_asal[i - 1] + hist_asal[i]
        cdf_target[i] = cdf_target[i - 1] + hist_target[i]

    cdf_asal = cdf_asal / cdf_asal[-1]
    cdf_target = cdf_target / cdf_target[-1]

    map_hist = np.zeros(256, dtype=np.uint8)

    # mapping histogram
    for i in range(256):

        selisih = np.abs(cdf_asal[i] - cdf_target)
        map_hist[i] = np.argmin(selisih)

    height, width = citra_asal.shape
    hasil = np.zeros((height, width), dtype=np.uint8)

    # terapkan mapping
    for y in range(height):
        for x in range(width):
            hasil[y, x] = map_hist[citra_asal[y, x]]

    return hasil


def ekualisasi_histogram(citra):
    import numpy as np

    height, width = citra.shape

    hist = np.zeros(256, dtype=int)

    # hitung histogram citra
    for y in range(height):
        for x in range(width):
            pixel = citra[y, x]
            hist[pixel] += 1

    cdf = np.zeros(256, dtype=int)
    cdf[0] = hist[0]

    # hitung CDF
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # normalisasi CDF
    cdf_normal = np.round(cdf * 255 / (height * width)).astype(np.uint8)

    hasil = np.zeros_like(citra, dtype=np.uint8)

    # terapkan hasil CDF normalisasi ke citra
    for y in range(height):
        for x in range(width):
            pixel = citra[y, x]
            hasil[y, x] = cdf_normal[pixel]

    return hasil

def buat_hist(citra): 
    histogram = [0] * 256 

    height = len(citra) 
    width = len(citra[0]) if height > 0 else 0 
    for i in range(height): 
        for j in range(width): 
            val = int(citra[i][j])   
            histogram[val] += 1   

    return histogram 

def ekualisasi_rgb(citra):
    import numpy as np

    # pisahkan channel BGR
    b = citra[:, :, 0]
    g = citra[:, :, 1]
    r = citra[:, :, 2]

    def equal_channel(channel):

        height, width = channel.shape

        # histogram
        hist = np.zeros(256, dtype=int)

        for y in range(height):
            for x in range(width):
                pixel = channel[y, x]
                hist[pixel] += 1

        # CDF
        cdf = np.zeros(256, dtype=int)
        cdf[0] = hist[0]

        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + hist[i]

        # normalisasi
        cdf_normal = np.round(
            cdf * 255 / (height * width)
        ).astype(np.uint8)

        # hasil channel
        hasil = np.zeros_like(channel, dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                pixel = channel[y, x]
                hasil[y, x] = cdf_normal[pixel]

        return hasil

    # equalization tiap channel
    b_eq = equal_channel(b)
    g_eq = equal_channel(g)
    r_eq = equal_channel(r)

    # gabungkan lagi channel
    hasil = np.zeros_like(citra)

    hasil[:, :, 0] = b_eq
    hasil[:, :, 1] = g_eq
    hasil[:, :, 2] = r_eq

    return hasil

def plot_histogram(histogram, title, ImgColor): 
    plt.figure(figsize=(10, 5)) 
    plt.xlabel("Intensitas Piksel") 
    plt.title(title) 
    plt.ylabel("Jumlah Piksel") 
    plt.bar(range(256), histogram, color=ImgColor, width=0.8) 
    plt.show()

def change_bg(img1, img2):
    hasil = np.zeros_like(img1, dtype=int) 
    for i in range(hasil.shape[0]): 
        for j in range(hasil.shape[1]): 
            if(img1[i,j]>50): 
                hasil[i,j] = img2[i,j] 
    # plt.imshow(hasil, cmap="gray") 
    return hasil

def masking(gray, bg):

    h, w = gray.shape
    hb, wb = bg.shape[:2]

    result = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):

            pixel = gray[y, x]

            # looping background
            by = y % hb
            bx = x % wb
            if pixel > 245:

                result[y, x, 0] = bg[by, bx, 0]
                result[y, x, 1] = bg[by, bx, 1]
                result[y, x, 2] = bg[by, bx, 2]

            else:
                result[y, x, 0] = pixel
                result[y, x, 1] = pixel
                result[y, x, 2] = pixel

    return result