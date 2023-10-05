import numpy as np
import dlib
import cv2
import glob
from PIL import Image
import os 

## Apply face detection and pre-processing techniques
def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

  ## Detect faces
  detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
  rects = detector(image, 1)

  w_max = 0
  h_max = 0

  found_face = False
  for faceRect in rects:
    rect = faceRect.rect
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    if (w*h)>(w_max*h_max):
      w_max=w
      h_max=h
      found_face = True
      ## crop the bigger face in the image
      img_crop = image[abs(rect.top()):abs(rect.bottom()), abs(rect.left()):abs(rect.right())]

  if found_face:
      img_crop = Image.fromarray(np.uint8(img_crop))
      resized_crop = img_crop.resize((48, 48), Image.ANTIALIAS)
      data_crop = np.asarray(resized_crop, dtype="uint8").reshape([48,48])
  else:
      return None

  data_crop = data_crop - np.mean(data_crop)
  f_b = np.sqrt(np.sum(np.square(data_crop)))

  if f_b==0:
    return None

  data_crop = data_crop * (100 / f_b)

  return data_crop

## To save numpy files from images
def save_data(X_train, y_train, fname='', folder='../preprocess_data/'):
    np.save(folder + 'X_' + fname, X_train)
    np.save(folder + 'Y_' + fname, y_train)

## To convert FER data from a CSV file into images
def convert_csv_to_img(path):

    with open(path) as f:
        content = f.readlines()

    lines = np.array(content)

    num_of_instances = lines.size

    for i in range(1, num_of_instances):
        try:
            emotion, img, split = lines[i].split(",")
            img = img.replace('"', '')
            img = img.replace('\n', '')
            pixels = img.split(" ")

            pixels = np.array(pixels, 'float32')
            image = pixels.reshape(48, 48)

            path_file_name = str(i)+"_example.jpg"
            cv2.imwrite(path_file_name, image)

        except Exception as ex:
            print(ex)

if __name__ == "__main__":

    print(">>>preprocessing<<<")
    
    # Read the CSV file containing the image file names and their corresponding labels
    with open("../data/image_labels.csv", "r") as f:
        lines = f.readlines()[1:]  # Skip the header row

    image_labels = {}
    for line in lines:
        file_name, label = line.strip().split(",")
        image_labels[file_name] = int(label)
        
    ####### Train 		
    cv_img = []
    cv_label = []

    for img in glob.glob("../data/Train/*.jpg"):
        print(img)
        input = cv2.imread(img)
        #print(input)
        file_name = os.path.basename(img)
        print(file_name)
        detected_input = format_image(input)
        if detected_input is None:
            continue
        label = image_labels[file_name]
        cv_label.append(label)
        cv_img.append(detected_input)

    ## Save in numpy files
    save_data(cv_img, cv_label, 'train')

    ####### Validate 
    cv_img = []
    cv_label = []

    for img in glob.glob("../data/Validate/*.jpg"):
        input = cv2.imread(img)
        file_name = os.path.basename(img)
        detected_input = format_image(input)
        if detected_input is None:
            continue
        label = image_labels[file_name]
        cv_label.append(label)
        cv_img.append(detected_input)

    ## Save in numpy files
    save_data(cv_img, cv_label, 'Validate')

    ###### Test 
    cv_img = []
    cv_label = []
    
    for img in glob.glob("../data/Test/*.jpg"):
        input = cv2.imread(img)
        file_name = os.path.basename(img)
        detected_input = format_image(input)
        if detected_input is None:
            continue
        label = image_labels[file_name]
        cv_label.append(label)
        cv_img.append(detected_input)

    ## Save in numpy files
    save_data(cv_img, cv_label, 'Test')
    
    ###### Final Test

    cv_img = []
    cv_label = []
    
    for img in glob.glob("../data/Test_final/*.jpg"):
        input = cv2.imread(img)
        file_name = os.path.basename(img)
        detected_input = format_image(input)
        if detected_input is None:
            continue
        label = image_labels[file_name]
        cv_label.append(label)
        cv_img.append(detected_input)

    ## Save in numpy files
    save_data(cv_img, cv_label, 'Test_final')