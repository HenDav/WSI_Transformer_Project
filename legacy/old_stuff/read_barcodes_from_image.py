import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar import pyzbar
from PIL import Image
from re import search
from pylibdmtx.pylibdmtx import decode as decode
import matplotlib.patches as patches
# Load image, grayscale, Gaussian blur, Otsu's threshold
img_path = r'C:\ran_data\RAMBAM\SlideID_images\box 1_1.png'
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#all_codes = decode(image)
all_codes = decode(gray)
#all_codes = decode(thresh)

gray_pil = Image.fromarray(gray)
w,h = gray.shape
gray_pil2 = gray_pil.resize((h * 2, w * 2), resample=Image.BICUBIC)
all_codes = decode(np.array(gray_pil2))

image = np.array(gray_pil2)
h_img = image.shape[0]
fig, ax = plt.subplots(1)
ax.imshow(image)
for code in all_codes:
    rect = patches.Rectangle((code.rect.left, h_img-code.rect.top-code.rect.height), code.rect.width, code.rect.height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

#ROI = gray[2170:2250,2225:2325] #doesn't work on phone
ROI = gray[1460:1540, 2215:2300] #works
plt.figure()
plt.imshow(ROI)
code1 = decode(ROI)

'''
#margin = 0.2
w,h = ROI.shape
#ROI = original[y - int(h * margin):y + int(h * (1+margin)), x - int(w * margin):x + int(w * (1+margin))]
ROI_pil = Image.fromarray(ROI)
#ROI_pil2 = ROI_pil.resize((h * 5, w * 5), resample=Image.BICUBIC)
ROI_pil2 = ROI_pil.resize((h * 5, w * 5), resample=Image.LANCZOS)
thresh = cv2.threshold(np.array(ROI_pil2), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
plt.figure()
plt.imshow(thresh)
'''
#barcode1 = pyzbar.decode(ROI_pil2, symbols=[ZBarSymbol.QRCODE])
barcode1 = decode(thresh)

######################################################
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9,9), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#plt.imshow(thresh)

# Morph close
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

#plt.imshow(close)
#plt.show()

# Find contours and filter for QR code
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#drawContours = cv2.drawContours(image, cnts, -1, (0, 0, 255), -1)
#cv2.imshow("Contours", drawContours)
#cv2.waitKey()

cnts = cnts[0] if len(cnts) == 2 else cnts[1]
'''ROI_count = 0
all_barcodes = []
all_area = []
all_ar = []
all_approx = []
all_peri = []'''
result_list = []
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    area = cv2.contourArea(c)
    ar = w / float(h)
    #if len(approx) == 4 and area > 1000 and (ar > .85 and ar < 1.3):
    if len(approx) == 4 and area > 500:
    #if True:
    #if len(approx) == 4 and(ar > .85 and ar < 1.3):
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
        #ROI = original[y:y+h, x:x+w]
        #ROI = original[y - int(h * 0.05):y + int(h * 1.05), x - int(w * 0.05):x + int(w * 1.05)]
        #ROI = thresh[y - int(h * 0.05):y + int(h * 1.05), x - int(w * 0.05):x + int(w * 1.05)]
        #ROI_count += 1
        #

        use_qr = True

        # plt.show()
        if not use_qr:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = r'C:\ran_programs\Tesseract-OCR\tesseract.exe'
            ROI = original[y:y + h, x:x + w]
            #plt.imshow(ROI)
            result = pytesseract.image_to_string(ROI)
            if len(result) > 3:
                print(result)
                if result[0] == '1' and result[2] == '-':
                    ROI2 = original[y-int(0.2*h):y+3*h, x-int(0.2*w):x+int(1.2*w)]
                    #plt.imshow(ROI2)
                    ROI2_pil = Image.fromarray(ROI2).resize((int(1.6*w*5), int(3.3*h*5)), resample=Image.BICUBIC)
                    #plt.imshow(ROI2_pil)

                    result2 = pytesseract.image_to_string(ROI2_pil)

                    match1 = search('\d{2}-\d{4}', result2)
                    #match1.group(0)
                    #match2 = search(r'\.{1}\.{1}\.{1}', result2)
                    #match2 = search('/\d{1}/\d{1}/\[a-z]{1}', result2)
                    match2 = search('/\d{1}/\d{1}/[a-z]', result2)
                    #match2.group(0)
                    if match1.group is not None and match2.group is not None:
                        result_final = match1.group(0) + ' ' + match2.group(0)
                    #result_final = result2.replace('\n',' ').replace(' \x0c', '')
                        result_list.append(result_final)
        else:
            #ROI = original[y - int(h * 0.1):y + int(h * 1.1), x - int(w * 0.1):x + int(w * 1.1)]
            #margin = 0.1
            margin = 0.2
            ROI = original[y - int(h * margin):y + int(h * (1+margin)), x - int(w * margin):x + int(w * (1+margin))]
            ROI_pil = Image.fromarray(ROI)
            ROI_pil2 = ROI_pil.resize((w * 3, h * 3))
            #plt.figure()
            plt.imshow(ROI_pil2)
            #barcode1 = pyzbar.decode(ROI_pil2, symbols=[ZBarSymbol.QRCODE])
            barcode1 = decode(ROI)
            #bw_im = np.array(binarization.nlbin(ROI_pil2))
            #bw_im = 255 - bw_im
            #plt.figure()
            #plt.imshow(bw_im)
            #barcode2 = pyzbar.decode(bw_im, symbols=[ZBarSymbol.QRCODE])
            print('aa')

        '''ROI_pil = Image.fromarray(ROI)
        #ROI_pil2 = ROI_pil.resize((w*10, h*10), resample=Image.BOX)
        ROI_pil2 = ROI_pil.resize((w * 16, h * 16))

        #tiling
        im = np.array(ROI_pil2)
        M = im.shape[0] // 16
        N = im.shape[1] // 16
        tiles = [im[x:x + M, y:y + N] for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)]
        tile_color = [np.mean(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)) for tile in tiles]
        im2 = np.zeros_like(im)
        count = 0
        for x in range(0, im2.shape[0], M):
            for y in range(0, im2.shape[1], N):
                im2[x:x + M, y:y + N] = tile_color[count]
                count+=1
        plt.imshow(im2)

        ROI_pil2 = cv2.GaussianBlur(np.array(ROI_pil2), (3, 3), 0)
        plt.imshow(ROI_pil2)

        ROI_deconv = lucy_richardson_deconv(np.array(ROI_pil2), n_iter=10)
        plt.figure()
        plt.imshow(ROI_deconv)

        plt.imshow(original)
        q1 = lucy_richardson_deconv(original, n_iter=10)
        plt.figure()
        plt.imshow(q1)
        #gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        #blur1 = cv2.GaussianBlur(gray1, (3, 3), 0)
        #thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        #plt.imshow(thresh1)
        cv2.imwrite('ROI' + str(ROI_count) + '.png', ROI)

        barcode1 = pyzbar.decode(ROI_pil2)
        if barcode1:
            all_barcodes.append(barcode1)
            all_area.append(area)
            all_ar.append(ar)
            all_approx.append(approx)
            all_peri.append(peri)'''
print('aa')


'''cv2.imshow('thresh', thresh)
cv2.imshow('close', close)
cv2.imshow('image', image)
cv2.imshow('ROI', ROI)
cv2.waitKey()     '''