# USAGE
# python sliding_window.py --image images/br-street-sweep_10.mat 
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.flip(image,0)