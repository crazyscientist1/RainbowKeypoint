from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw
import numpy as np
import tensorflow as tf
import math
import random

def transform_image(image, size, angle=0, scale=1.0, shear=0, translate=(0, 0), enhance=1,blur=0):
    if scale != 1.0:
        width, height = size, size
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        image = Image.new("RGB", (width, height))
        image.paste(resized_image, ((width - new_width) // 2, (height - new_height) // 2))

    if angle != 0:
        image = image.rotate(angle)

    if shear != 0:
        width, height = size, size
        shear_matrix = (1, shear, 0, 0, 1, 0)
        image = image.transform((width, height), Image.AFFINE, shear_matrix)

    if translate != (0, 0):
        translation_matrix = (1, 0, translate[0], 0, 1, translate[1])
        image = image.transform(image.size, Image.AFFINE, translation_matrix)
    if enhance != 1:
      image = ImageEnhance.Color(image).enhance(enhance)

    if blur != 0:
      image = image.filter(ImageFilter.GaussianBlur(blur))

    return np.array(image)

def rotation(x,y, theta):
    theta *= -1
    hyp = math.sqrt(y ** 2 + x ** 2)
    currAngle = math.atan(y/x)
    newAngle = currAngle + theta * math.pi / 180

    yN = math.sin(newAngle) * hyp
    xN = math.cos(newAngle) * hyp

    if x >= 0 and y >= 0:
        return xN , yN
    elif x <= 0 and y >= 0:
        return -1 * xN , -1 * yN
    elif x < 0 and y < 0:
        return -1 * xN , -1 * yN
    else:
        return xN , yN

def pointTransform(stackedCoords, size, angle = 0, scale = 1, shear = 0, translate = (0,0)):
  coords = []
  for x, y in stackedCoords:
    if rotation != 0:
      x,y = rotation(x - size/2, y - size/2, angle)

    if scale != 1:
      x *= scale
      y *= scale

    if shear != 0:
      x = x - shear * (y + size/2)

    if translate != (0,0):
      x -= translate[0]
      y -= translate[1]

    x = (x +size/2)/size
    y = (y + size/2)/size
    coords.append([x,y])
  return coords

def augmenter(image, size, coords=0):

  angle = random.randint(0,360)
  shear = random.randint(-2000,2000)/10000
  enhance = random.randint(5000,15000)/10000
  blur = random.randint(5000,15000)/10000
  scale = random.randint(9000,11000)/10000
  translate = (random.randint(-15,15), random.randint(-15,15))


  if coords != 0:
    coords = pointTransform(coords, size=size, scale=scale,angle=angle, shear=shear,translate = translate)
    image = transform_image(image.copy(), size, scale=scale,angle=angle, shear=shear, enhance=enhance, blur = blur, translate = translate)
    return image, coords
  else:
    image = transform_image(image.copy(), size, scale=scale,angle=angle, shear=shear, enhance=enhance, blur = blur, translate = translate)
    return image
