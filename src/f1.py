import cv2

image = cv2.imread("data/raw-890/2_img_.png")

print(image.shape)

blue = image[:, :, 0]
green = image[:, :, 1]
red = image[:, :, 2]

print("Blue:", blue[100, 100])
print("Green:", green[100, 100])
print("Red:", red[100, 100])