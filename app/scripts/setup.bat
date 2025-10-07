# Test if everything works now
python -c "
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image, ImageDraw

print('Testing OCR engines...')

# Test image
img = Image.new('RGB', (400, 100), color='white')
draw = ImageDraw.Draw(img)
draw.text((20, 30), 'Aadhaar: 1234 5678 9012', fill='black')

# Test Tesseract
text1 = pytesseract.image_to_string(img)
print(f'Tesseract: {text1.strip()}')

# Test PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
result = ocr.ocr(np.array(img))
if result and result[0]:
    text2 = ' '.join([line[1][0] for line in result[0]])
    print(f'PaddleOCR: {text2}')

print('✅ Both OCR engines working!')
"