import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

def image_transform(image_np, linear_transform):
  # Get the dimensions of the image
  height, width, channels = image_np.shape

  # Define the center
  center_x = width / 2
  center_y = height / 2

  # Loop through each pixel in the image and apply the transformation
  transformed_image = np.zeros_like(image_np)

  for y in range(height):
      for x in range(width):
          # Translate the pixel to the origin
          translated_x = x - center_x
          translated_y = -(y - center_y)
          
          # Apply the transformation: matrix vector multiplication
          transformed_x, transformed_y = linear_transform@np.array([translated_x, translated_y])
          
          # Translate the pixel back to its original position
          transformed_x += center_x
          transformed_y = - transformed_y + center_y
          
          # Round the pixel coordinates to integers
          transformed_x = int(round(transformed_x))
          transformed_y = int(round(transformed_y))
          
          # Copy the pixel to the transformed image
          if (transformed_x >= 0 and transformed_x < width and transformed_y >= 0 and transformed_y < height):
              transformed_image[transformed_y, transformed_x] = image_np[y, x]

  return transformed_image


image_path = "jin_stanky_leg.png"
image0 = Image.open(image_path)
# image0.show()
plt.imshow(image0)
# Convert the image as a numpy array
image0_np = np.array(image0)
print(f'The dimension of image0 is {image0_np.shape}')


#transform 1: scale image0 by half, find stdmat of it and "print" it
# Define the transformation matrix
print("Transform 1: scale the original image by half, into image1")
mat1 = np.array([
    [0.5, 0],
    [0, 0.5]
])
# apply transform
image1_np = image_transform(image0_np, mat1)
print(f'The transform matrix is: {mat1}\n')
# convert a numpy array to image
image1 = Image.fromarray(image1_np)
#image1.show()
plt.imshow(image1)
image1.save("transform1.png", "PNG")

#transform 2: reflect img1 over y=2x. print stdmat and image2
print("Transform 2: reflect image1 over y=2x, into image2")
rtan = 2*(math.atan(2))
rcos = math.cos(rtan)
rsin = math.sin(rtan)
mat2 = np.array([
    [rcos, rsin],
    [rsin, -rcos]
])
# apply transform
image2_np = image_transform(image1_np, mat2)
print(f'The transform matrix is: {mat2}\n')
# convert a numpy array to image
image2 = Image.fromarray(image2_np)
#image2.show()
plt.imshow(image2)
image2.save("transform2.png", "PNG")

#transform 3: reflect img2 over y=-1/2x -> img3
print("Transform 3: reflect image2 over y=-0.5x, into image3")
rtan = 2*(math.atan(-0.5))
rcos = math.cos(rtan)
rsin = math.sin(rtan)
mat3 = np.array([
    [rcos, rsin],
    [rsin, -rcos]
])
# apply transform
image3_np = image_transform(image2_np, mat3)
print(f'The transform matrix is: {mat3}\n')
# convert a numpy array to image
image3 = Image.fromarray(image3_np)
#image3.show()
plt.imshow(image3)
image3.save("transform3.png", "PNG")

#transform 4: find mat of T2 * T1, apply to img1
print("Transform 4: composition of T2 and T1, apply to image1")
#mat = np.array([[-0.3, 0.4],[0.4, 0.3]])
mat4 = np.multiply(mat2, mat1)
imageT_np = image_transform(image1_np, mat4)
print(f'The transform matrix is: {mat4}\n')
imageT = Image.fromarray(imageT_np)
#imageT.show()
plt.imshow(imageT)
imageT.save("transform4.png", "PNG")

#transform 5: invert transform 4, apply to img3
print("Transform 5: apply the inverse of T4 to image3")
mat5 = np.linalg.inv(mat4)
imageTI_np = image_transform(image3_np, mat5)
print(f'The transform matrix is: {mat5}\n')
imageTI = Image.fromarray(imageTI_np)
#imageTI.show()
plt.imshow(imageTI)
imageTI.save("transform5.png", "PNG")

#transform 6: favorite transform, apply to img1
print("Transform 6: my favorite transform, total randomness, into image4")
mat6 = np.array([
    [random.random(), random.random()],
    [random.random(), random.random()]
])
image4_np = image_transform(image1_np, mat6)
print(f'The transform matrix is: {mat6}\n')
image4 = Image.fromarray(image4_np)
#image4.show()
plt.imshow(image4)
image4.save("transform6.png", "PNG")
