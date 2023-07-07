import cv2
import numpy as np
import imutils
from imutils.contours import sort_contours

class RequestImageConverter:
    def __init__(self, file):
        self.file = file
    
    def convert(self) :
        numpy_image = np.fromstring(self.file, dtype='uint8')
        image = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)
        return image

class TextRecognizer:
    def __init__(self, image):
        self.image = image
        self.characters = []

    def recognize_text(self):
        '''
        STEP 1: IMAGE PREPROCESSING
        '''
        # print(self.image.shape) #n,n,3
        # parameters
        identitiy_matrix_shape = (3, 3) #used in dilatation (prev 3 3)

        # make the image color grey
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # n,n
        
        # invert the image color
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # dilatate the image
        dilated = cv2.dilate(thresh, np.ones(identitiy_matrix_shape))

        edges = cv2.Canny(dilated, 40, 150)

        # dilatate the image
        processed_image = cv2.dilate(edges, np.ones(identitiy_matrix_shape))
        # processed_image = cv2.dilate(edges, np.ones((4,)))

        '''
        STEP 2: TEXT RECOGNIZER
        SEGMENTATION
        '''
        # parameters
        min_w, max_w = 15, 1200 # sebelumnya 15. Untuk gambar ukuran 240, titik bisa dihilangkan dengan min w/h = 30. (update): dari 30 balik lagi jadi 15, karena mempertimbangkan lebar angka 1
        min_h, max_h = 30, 1200 # mula-mula 30
        
        # find countour
        conts = self.contour_detection(processed_image.copy())
        
        if not len(conts):
            return ''

        # sort the contour
        if len(conts) > 1:
            conts = self.sort_contour(conts)

        # prepare the output
        for box in conts:
            (x, y, w, h) = box
            if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
                # which image will be used to be the baseline of the prediction? ps: it must be the inverted one.
                self.process_box(thresh, x, y, w, h)
        
        pixels = np.array([pixel for pixel in self.characters], dtype = 'float32')

        return pixels

    # Detect each characater in the image
    def contour_detection(self, img):
        conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        conts = np.array([cv2.boundingRect(i) for i in conts])
        return conts

    # Extract Range of Interest (ROI)
    def extract_roi(self, conts, x, y, w, h):
        roi = conts[y:y + h, x:x + w]
        return roi

    # Resize the Image
    def resize_img(self, img, w, h):
        if w > h:
            resized = imutils.resize(img, width = 28)
        else:
            resized = imutils.resize(img, height = 28)

        (h, w) = resized.shape
        dX = int(max(0, 28 - w) / 2.0)
        dY = int(max(0, 28 - h) / 2.0)

        filled = cv2.copyMakeBorder(resized, top=dY, bottom=dY, right=dX, left=dX, borderType=cv2.BORDER_CONSTANT, value = (0,0,0))
        filled = cv2.resize(filled, (28,28))
        return filled

    # normalize the image, and expand the dimension so it click our model dims
    def normalization(self, img):
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis = -1)
        return img

    def process_box(self, img, x, y, w, h):
        roi = self.extract_roi(img, x, y, w, h)
        resized = self.resize_img(roi, w, h)
        normalized = self.normalization(resized)
        self.characters.append(normalized)

    def sort_contour(self, conts):
        # sort the countur, 1st top to bottom (line by line), then left to right for each line
        # sort the data from y values/top
        sort_by_line = conts[np.argsort(conts[:, 1])]

        # slice data to lines by the difference of every y where 
        # y is greater that median of the char heights
        median_h = np.median(sort_by_line[:, -1])
        diff_y = np.diff(sort_by_line[:,1])
        new_line = np.where(diff_y > median_h-5)[0] + 1 #np.where value need to be adjust to make the code predict line by line better
        lines = np.array_split(sort_by_line, new_line)

        # sorted each lines from left.
        sorted_left = [line[np.argsort(line[:, 0])] for line in lines]

        sorted_box = [box for lines in sorted_left for box in lines]

        return sorted_box