import os
from flask import Flask


def pca_image_dir(args):
   import cv2
   import numpy as np

   in_path = args.images_directory
   stride = args.stride
   target_mean_path = args.target_mean
   target_mean = None
   if target_mean_path != None:
       target_mean = np.load(target_mean_path)
       print "TARGET_MEAN shape: " + str(target_mean.shape) + ", type: " + str(target_mean.dtype)

   files = os.listdir(in_path)
   out_dir = os.path.abspath(in_path) + "_every-" + str(stride) + "_PCA"

   f = 0
   matrix_test = None
   imgraw = None
   print " "
   for image in files:
       f = f + 1
       if f % stride != 0:
	   continue
       imgraw = cv2.imread(os.path.join(in_path, image), -1)
       if target_mean != None:
           imgraw = crop_resize_convert_to_match_target_image(imgraw, target_mean)
       imgvector = imgraw.reshape(imgraw.size)
       #print imgvector
       try:
	   matrix_test = np.vstack((matrix_test, imgvector))
       except:
	   matrix_test = imgvector
       print str(f) + " of " + str(len(files)) + "\r",
   print ""

   # PCA
   print "Running PCA ..."
   mean, eigenvectors = cv2.PCACompute(matrix_test, np.mean(matrix_test, axis=0).reshape(1,-1))

   mean = mean.reshape(imgraw.shape)
   #print "mean: " + str(mean)
   print " "

   os.mkdir(out_dir)

   mean_filepath = os.path.join(out_dir, "mean.npy")
   np.save(mean_filepath, mean)

   meanimg = mean
   #meanimg *= 255.0/mean.max()
   cv2.imwrite(os.path.join(out_dir, "mean.png"), meanimg)

   i = 0
   for eigenvector in eigenvectors:
       filename = "%05d" % i
       path_no_extension = os.path.join(out_dir, filename)
       npy_path = path_no_extension + ".npy"
       cvxml_path = path_no_extension + ".xml"
       eigenvector = eigenvector.reshape(imgraw.shape)
       np.save(npy_path, eigenvector)
   #    print "Saved " + path
   #    print eigenvector
   #    print "--------------------------"
   #    print "min: " + str(evimg.min()) + ", max: " + str(evimg.max())
   #    evimg += evimg.min()
       evimg = eigenvector
       evimg *= 255.0/eigenvector.max()
       cv2.imwrite(os.path.join(out_dir, filename + ".png"), evimg)
       i = i + 1

   print "Saved " + str(len(eigenvectors)) + " eigenvectors to " + out_dir


app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
