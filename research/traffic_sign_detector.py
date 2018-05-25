import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob
from datetime import datetime

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = 'models/traffic_sign/frozen_graphs/frozen_inference_graph.pb'
PATH_TO_CKPT = 'exported_graphs/gtsdb/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'gtsrb_traffic_sign_label_map.pbtxt')

NUM_CLASSES = 43


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images/gtsdb_test/'
PATH_TO_LABELLED_TEST_IMAGES_DIR = '/Users/liangchuangu/Downloads/TrainIJCNN2013/labelled'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.png'.format(i)) for i in range(1, 3) ]
#TEST_IMAGE_PATHS = glob.glob('object_detection/test_images/gtsdb_test/*.jpg')
TEST_IMAGE_PATHS = glob.glob('/Users/liangchuangu/Downloads/TrainIJCNN2013/*.jpg')

prohibitory = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
mandatory = {33, 34, 35, 36, 37, 38, 39, 40}
danger = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    fp = open(os.path.join(PATH_TO_LABELLED_TEST_IMAGES_DIR, 'ex_p.txt'),'w')
    fm = open(os.path.join(PATH_TO_LABELLED_TEST_IMAGES_DIR, 'ex_m.txt'),'w')
    fd = open(os.path.join(PATH_TO_LABELLED_TEST_IMAGES_DIR, 'ex_d.txt'),'w')
    fo = open(os.path.join(PATH_TO_LABELLED_TEST_IMAGES_DIR, 'ex_o.txt'),'w')
    counter = 0
    total = len(TEST_IMAGE_PATHS)
    for image_path in TEST_IMAGE_PATHS:
      start=datetime.now()
      counter = counter + 1
      image = Image.open(image_path)
      width, height = image.size
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      #print(*boxes, sep=' -> ')
      #print(*scores, sep=' | ')
      #print(*classes, sep=', ')
      #print('number: ' + str(num))
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=10,
          min_score_thresh=.6,
          line_thickness=4)
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)
      #plt.show()
      print("{0:.2f}%".format(counter/total * 100) + "% #============ Labelling image " + str(counter) + " over " + str(total) + " images ============#")
      predicted_indexes = []
      for idx, confidence in enumerate(scores[0]):
        if confidence > 0.1:
          predicted_indexes.append(idx)
          print(str(idx + 1) + " confidence: " + str(confidence))
        else:
          break
      file_name = image_path.split('/')[-1]
      print("Detected " + str(len(predicted_indexes)) + " traffic signs in image " + file_name)
      name_as_ppm = file_name.split('.')[0] + ".ppm"
      c = int(classes[0][idx]) - 1
      if c < 0:
        print("Negative class: " + str(c))
        exit(-1)
      if len(predicted_indexes) > 0:
        for idx in predicted_indexes:
          box_str = str(int(boxes[0][idx][1]*width)) + ";" + str(int(boxes[0][idx][0]*height)) + ";" + str(int(boxes[0][idx][3]*width)) + ";" + str(int(boxes[0][idx][2]*height))
          if c in prohibitory:
            fp.write(name_as_ppm + ";" + box_str + "\n")
          elif c in mandatory:
            fm.write(name_as_ppm + ";" + box_str + "\n")
          elif c in danger:
            fd.write(name_as_ppm + ";" + box_str + "\n")
          else:
            fo.write(name_as_ppm + ";" + box_str + "\n")
          print(" [" + box_str + ";" + str(int(classes[0][idx])) + "] ")
      #print("Labelling for " + image_path)
      Image.fromarray(image_np).save(os.path.join(PATH_TO_LABELLED_TEST_IMAGES_DIR, image_path.split('/')[-1]))
      print("#============ Elapsed time: " + str((datetime.now()-start).total_seconds() * 1000) + " ms ============#\n")
    fp.close()
    fm.close()
    fd.close()
    fo.close()
