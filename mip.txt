!pip install numpy
!pip install matplotlib 
!pip install opencv-python

import numpy as np
print(np.__version__)
import matplotlib.pyplot as plt
import cv2
print(cv2.__version__)  
import math

img=cv2.imread("image.jpg",0) 
res=cv2.imread ("imgrgb.jpg",0)
gray=img
plt.imshow(img, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

plt.imshow(img_rgb, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show() 


img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

plt.imshow(img_g, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show() 


grayscale = np.array(img_rgb)
gn=grayscale/255
print(gn)

plt.imshow(gn, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

def nearest_neighbor_resize(img, new_h, new_w):
    h, w = img.shape
    resized = np.zeros((new_h, new_w), dtype=img.dtype)

    row_scale = h / new_h
    col_scale = w / new_w

    for i in range(new_h):
        for j in range(new_w):
            src_i = round(i * row_scale)
            src_j = round(j * col_scale)

            src_i = min(src_i, h - 1)
            src_j = min(src_j, w - 1)

            resized[i, j] = img[src_i, src_j]

    return resized

h, w = img.shape

zoom_in  = nearest_neighbor_resize(img, int(2*h), int(2*w))

plt.imshow(zoom_in, cmap='gray', aspect='equal')
plt.title("Grayscale Image")
#plt.axis('off')
plt.show()

print(zoom_in.shape)

zoom_out = nearest_neighbor_resize(img, int(0.5*h), int(0.5*w)) 

plt.imshow(zoom_out, cmap='gray', aspect='equal')
plt.title("Grayscale Image")
#plt.axis('off')
plt.show() 

print(zoom_out.shape)

def bilinear_resize(img, new_h, new_w):
    h, w = img.shape
    resized = np.zeros((new_h, new_w), dtype=np.float32)

    row_scale = (h - 1) / (new_h - 1)
    col_scale = (w - 1) / (new_w - 1)

    for i in range(new_h):
        for j in range(new_w):

            x = i * row_scale
            y = j * col_scale

            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            x2 = min(x1 + 1, h - 1)
            y2 = min(y1 + 1, w - 1)

            dx = x - x1
            dy = y - y1

            val = (
                img[x1, y1] * (1 - dx) * (1 - dy) +
                img[x2, y1] * dx * (1 - dy) +
                img[x1, y2] * (1 - dx) * dy +
                img[x2, y2] * dx * dy
            )

            resized[i, j] = val

    return resized.astype(img.dtype)

h, w = gray.shape

zoom_in  = bilinear_resize(gray, int(2*h), int(2*w))
zoom_out = bilinear_resize(gray, int(0.5*h), int(0.5*w))

import matplotlib.pyplot as plt

plt.figure(figsize=(18,12))

plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Original")
#plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(zoom_in, cmap='gray')
plt.title("Bilinear Zoom In")
#plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(zoom_out, cmap='gray')
plt.title("Bilinear Zoom Out")
#plt.axis('off')

plt.show()

subtraction = np.clip(img - res, 0, 255).astype(np.uint8)

plt.imshow(subtraction)
plt.title("Subtraction Image")
plt.axis("off")
plt.show()

def gray_level_resolution(img, bits):
    L = 2 ** bits         
    step = 256 // L       

    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            level = img[i, j] // step
            out[i, j] = level * step

    return out

reduced_4bit = gray_level_resolution(gray, 4)
reduced_2bit = gray_level_resolution(gray, 1) 

plt.figure(figsize=(18,12))

plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Original (8-bit)")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(reduced_4bit, cmap='gray')
plt.title("4-bit (16 levels)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(reduced_2bit, cmap='gray')
plt.title("2-bit (2 levels)")
plt.axis('off')

plt.show()

#NEGATIVE TRANSFORMATION
image_neg = 255 - img 
plt.imshow(image_neg,cmap='gray')
plt.title("Negative Image")
plt.axis("off")
plt.show()

#Log
image_float = np.float32(img)
c = 255/math.log10(256)
image_log = c*np.log10(1+image_float)
image_log = np.uint8(image_log)

plt.imshow(image_log,cmap='gray')
plt.title("Logarithmic Transform")
plt.axis("off")
plt.show()

#POWER?GAMMA TRANSFORMATION

image_norm = img/255
gamma = 0.1
c = 255
image_gamma = c*np.power(image_norm, gamma)
image_gamma = np.uint8(image_gamma) 

plt.imshow(image_gamma, cmap='gray')
plt.title("Power Transform")
plt.axis("off")
plt.show()

#CONTRAST STRETCHING
def piecewise_contrast_loop(img, a, b, alpha, beta, gamma):
    L = 256

    img = img.astype(np.float32)
    h, w = img.shape

    out = np.zeros((h, w), dtype=np.float32)

    # Continuity constants
    Sa = alpha * a
    Sb = beta * (b - a) + Sa

    for i in range(h):
        for j in range(w):

            r = img[i, j]

            # Region 1: 0 ≤ r ≤ a
            if r <= a:
                s = alpha * r

            # Region 2: a < r ≤ b
            elif r <= b:
                s = beta * (r - a) + Sa

            # Region 3: b < r ≤ L-1
            else:
                s = gamma * (r - b) + Sb

            # Clamp to valid range
            if s < 0:
                s = 0
            elif s > 255:
                s = 255
            out[i, j] = s
    return out.astype(np.uint8)

a = 60
b = 180

alpha = 0.5   # compress darks
beta  = 1.8   # stretch mids
gamma = 0.5   # compress highlights

result = piecewise_contrast_loop(gray, a, b, alpha, beta, gamma)

plt.subplot(1,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(result, cmap='gray')
plt.title("Piecewise Contrast")
plt.axis('off')

plt.show()

# Parameters (same as your program)
a = 60
b = 180
alpha = 1
beta  = 1.8
gamma = 1.4

L = 256

# Continuity constants
Sa = alpha * a
Sb = beta * (b - a) + Sa

r = np.arange(0, L)
s = np.zeros_like(r, dtype=float)

for i in range(L):

    if r[i] <= a:
        s[i] = alpha * r[i]

    elif r[i] <= b:
        s[i] = beta * (r[i] - a) + Sa

    else:
        s[i] = gamma * (r[i] - b) + Sb

# Plot
plt.plot(r, s)
plt.title("Piecewise Linear Contrast Stretching")
plt.xlabel("Input Intensity (r)")
plt.ylabel("Output Intensity (s)")
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.grid(True)
plt.plot(r, r, '--', label='Identity (s=r)')
plt.plot(r, s, label='Transform')
plt.legend()
plt.show()

binary = np.zeros_like(gray)

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        if gray[i, j] >= 128:
            binary[i, j] = 255
        else:
            binary[i, j] = 0

plt.imshow(binary, cmap='gray')
plt.title("Threshold")
plt.axis('off')

plt.show()

bit_planes = []

for k in range(8):
    plane = (img >> k) & 1
    plane = plane * 255
    plane = plane.astype(np.uint8)
    bit_planes.append(plane) 

plt.figure(figsize=(12,6))

for k in range(8):
    plt.subplot(2,4,k+1)
    plt.imshow(bit_planes[k], cmap='gray')
    plt.title(f'Bit Plane {k}')
    plt.axis('off')

plt.tight_layout()
plt.show()

#gray slicing
def gray_slice_with_bg(img, A, B):
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):

            r = img[i, j]

            if A <= r <= B:
                out[i, j] = 255
            else:
                out[i, j] = 0

    return out

A = 80
B = 100

slice2 = gray_slice_with_bg(gray, A, B) 

plt.figure(figsize=(18,12))

plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(slice2, cmap='gray')
plt.title("Slicing")
plt.axis('off')

plt.show()





#hist eq
hist = np.zeros(256, dtype=int)
for pixel in image_log.flatten():
    hist[pixel] += 1 

x = np.arange(256)
plt.figure(figsize=(10,5))
plt.bar(x, hist, width=1.0, color='orange')
plt.title("Histogram")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")
plt.xlim([0,255])
plt.show()

def histogram_equalization(img):
    L = 256
    h, w = img.shape
    N = h * w

    hist = np.zeros(L, dtype=int)

    for i in range(h):
        for j in range(w):
            r = img[i, j]
            hist[r] += 1

    cdf = np.zeros(L, dtype=int)
    cdf[0] = hist[0]/N

    for i in range(1, L):
        cdf[i] = cdf[i-1] + hist[i]

    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):

            r = img[i, j]

            s = (L - 1) * cdf[r] / N

            out[i, j] = int(s)

    return out

equalized = histogram_equalization(image_log) 

plt.subplot(1,2,1)
plt.imshow(image_log, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(equalized, cmap='gray')
plt.title("Equalized")
plt.axis('off')

plt.show() 

plt.bar(x, hist, width=1.0, color='orange')
plt.title("Original Histogram")
plt.show()

plt.hist(equalized.ravel(), bins=256)
plt.title("Equalized Histogram")

plt.show()

#hist match
def histogram_specification(source, target):

    L = 256
    h, w = source.shape
    N = h * w

    hist_src = np.zeros(L, dtype=int)

    for i in range(h):
        for j in range(w):
            hist_src[source[i, j]] += 1

    cdf_src = np.zeros(L, dtype=float)
    cdf_src[0] = hist_src[0] / N

    for i in range(1, L):
        cdf_src[i] = cdf_src[i-1] + hist_src[i] / N


    ht, wt = target.shape
    Nt = ht * wt

    hist_tgt = np.zeros(L, dtype=int)

    for i in range(ht):
        for j in range(wt):
            hist_tgt[target[i, j]] += 1

    cdf_tgt = np.zeros(L, dtype=float)
    cdf_tgt[0] = hist_tgt[0] / Nt

    for i in range(1, L):
        cdf_tgt[i] = cdf_tgt[i-1] + hist_tgt[i] / Nt


    mapping = np.zeros(L, dtype=np.uint8)

    for r in range(L):

        # find z whose CDF is closest
        diff = abs(cdf_src[r] - cdf_tgt[0])
        z_best = 0

        for z in range(1, L):
            d = abs(cdf_src[r] - cdf_tgt[z])
            if d < diff:
                diff = d
                z_best = z

        mapping[r] = z_best


    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            out[i, j] = mapping[source[i, j]]

    return out

result = histogram_specification(image_log, img) 

plt.figure(figsize=(18,12))

plt.subplot(1,3,1)
plt.imshow(image_log, cmap='gray')
plt.title("Source")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(img, cmap='gray')
plt.title("Target")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(result, cmap='gray')
plt.title("Matched")
plt.axis('off')

plt.show()

h, w = img.shape
output = np.zeros_like(img)

tile_size=8
clip_limit=40

tile_h = h // tile_size
tile_w = w // tile_size

for i in range(tile_size):
    for j in range(tile_size):

        tile = img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]

        hist, bins = np.histogram(tile.flatten(), 256, [0,256])
        hist = np.minimum(hist, clip_limit)

        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf = cdf.astype('uint8')

        output[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = cdf[tile] 

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("CLAHE without binary interpolation")
plt.imshow(output, cmap="gray")
plt.axis("off")

plt.show()

all_cdfs = []
for i in range(tile_size):
    row_cdfs = []
    for j in range(tile_size):
        
        tile = img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
        
        
        hist, _ = np.histogram(tile.flatten(), 256, [0,256])
        limit = max(1, clip_limit * tile.size // 256)
        excess = np.sum(np.maximum(hist - limit, 0))
        hist = np.minimum(hist, limit) + (excess // 256)

        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
        row_cdfs.append(cdf.astype('uint8'))
        
    all_cdfs.append(row_cdfs)
    
    
output = np.zeros_like(img)

for y in range(h):
    for x in range(w):
        
        ty = (y - tile_h / 2) / tile_h
        tx = (x - tile_w / 2) / tile_w
        

        y1, x1 = int(np.floor(ty)), int(np.floor(tx))
        y2, x2 = y1 + 1, x1 + 1
        

        fy, fx = ty - y1, tx - x1
        

        y1, y2 = np.clip([y1, y2], 0, tile_size - 1)
        x1, x2 = np.clip([x1, x2], 0, tile_size - 1)
        

        v11 = all_cdfs[y1][x1][img[y, x]]
        v12 = all_cdfs[y1][x2][img[y, x]]
        v21 = all_cdfs[y2][x1][img[y, x]]
        v22 = all_cdfs[y2][x2][img[y, x]]
        

        val_top = (1 - fx) * v11 + fx * v12
        val_bot = (1 - fx) * v21 + fx * v22
        

        output[y, x] = (1 - fy) * val_top + fy * val_bot

# Visualization
plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(img, cmap="gray")
plt.subplot(1,2,2); plt.title("CLAHE with Interpolation"); plt.imshow(output, cmap="gray")
plt.show()

def add_salt_and_pepper(image, prob=0.05):

    output = np.copy(image)
    rnd = np.random.random(image.shape)
    output[rnd > (1 - prob/2)] = 255
    output[rnd < (prob/2)] = 0
    return output 

def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8) 

def manual_median_filter(image, kernel_size=11):
    h, w = image.shape
    pad = kernel_size // 2

    padded_img = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            window = padded_img[i : i + kernel_size, j : j + kernel_size]
            output[i, j] = np.median(window)
    return output

sp_noisy = add_salt_and_pepper(img, prob=0.05)
gauss_noisy = add_gaussian_noise(img, sigma=20)

denoised_sp = manual_median_filter(sp_noisy, kernel_size=3)
denoised_gauss = manual_median_filter(gauss_noisy, kernel_size=3) 

plt.figure(figsize=(16, 8))

plt.subplot(2, 5, 1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(2, 5, 2)
plt.title("Salt & Pepper Noise")
plt.imshow(sp_noisy, cmap="gray")
plt.axis("off")

plt.subplot(2, 5, 3)
plt.title("Gaussian Noise")
plt.imshow(gauss_noisy, cmap="gray")
plt.axis("off")

plt.subplot(2, 5, 6) 
plt.title("Median Filtered (S&P)")
plt.imshow(denoised_sp, cmap="gray")
plt.axis("off") 

plt.subplot(2, 5, 7) 
plt.title("Median Filtered (Gaussian)")
plt.imshow(denoised_gauss, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

def manual_convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    
    padded_img = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=float)
    
    for i in range(h):
        for j in range(w):

            window = padded_img[i : i + kh, j : j + kw]
            output[i, j] = np.sum(window * kernel)
            
    return output 

sizes = [1, 3, 5, 7, 9]
avg_images = []

for size in sizes:
    kernel = np.ones((size, size)) / (size * size)

    avg_image = manual_convolve(img, kernel).astype(np.uint8)
    avg_images.append(avg_image) 

plt.figure(figsize=(18, 10))


plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

for i, (size, filtered_img) in enumerate(zip(sizes, avg_images)):
    plt.subplot(2, 3, i + 2)
    plt.title(f"Average Filter: {size}x{size}")
    plt.imshow(filtered_img, cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()

Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

Gx = manual_convolve(img, Kx)
Gy = manual_convolve(img, Ky)

magnitude = np.sqrt(Gx**2 + Gy**2)
edge_img= np.clip(magnitude, 0, 255).astype(np.uint8) 

plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1) 
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2) 
plt.title("Edge Filter (Sobel)")
plt.imshow(edge_img, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()


