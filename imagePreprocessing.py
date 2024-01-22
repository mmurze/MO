import pytesseract
import cv2
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'


def text_identification(img):
    texts = []
    config = r'--oem 3 --psm 6' 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.jpg',gray)
    texts.append(pytesseract.image_to_string(gray,  lang = 'rus', config=config))
    
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cv2.imwrite('threshold_image.jpg',thresh1)
    texts.append(pytesseract.image_to_string(thresh1,  lang = 'rus', config=config))
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 3)
    cv2.imwrite('dilation_image.jpg',dilation)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    cropped_text =''
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Рисуем ограничительную рамку на текстовой области
        rect=cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Обрезаем область ограничительной рамки
        cropped = im2[y:y + h, x:x + w]
        cropped_text += pytesseract.image_to_string(cropped,  lang = 'rus', config=config)
        cv2.imwrite('rectanglebox.jpg',rect)
    texts.append(cropped_text)
    return texts
  
        
def find_data_in_text(str):
  datas = np.zeros(4)
  regExp = [
    r'\b[Жж][Ии][Ии]?[Рр](а|ы|ов)?(, г|,г)?\s?[—-]?\s?(от)?\s?[0-9]+[,.\s]?[0-9]',
    r'\bбелк(а|и|ов)?(, г|,г)?\s?[—-]?\s?(не менее)?\s?[0-9]+[,.\s]?[0-9]',
    r'\bуглевод(а|ы|ов)?(, г|,г)?\s?[—-]?\s?[0-9]+[,.\s]?[0-9]',
    r'[\d]+(.[\d]+)?\sккал'
  ]
  
  for i in range(4):
    finded_slice = re.search(regExp[i], str, re.IGNORECASE)
    if finded_slice:
      specific_numbers=re.search(r'\b[0-9]+[,.\s]?[0-9]', finded_slice[0], re.IGNORECASE)
      number = float(specific_numbers[0].replace(",","."))
      if i == 3:
        datas[i] = number
      else:
        datas[i] = number/ 10 if float(specific_numbers[0].replace(",",".")) > 10 else number

  return datas

def result(pic):
  texts = text_identification(cv2.imread("pics/" + pic))
  data = [0, 0, 0, 0]
  flag = 0
  for text in texts:
      image_data = find_data_in_text(text)
      for i in range(4):
        if(data[i] == 0 and image_data[i] != 0):
          flag += 1
          data[i] = image_data[i]
      if flag == 4:
        break
  return data
