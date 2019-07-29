import numpy as np
import tensorflow as tf
import pygame
import os

if not os.path.isfile("tutorial1.tflite"):
    print("tutorial1.tflite not found.  Download a pretrained one with:")
    print("  wget https://hypergan.s3-us-west-1.amazonaws.com/0.10/tutorial1.tflite")
    exit(-1)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="tutorial1.tflite")
interpreter.allocate_tensors()

def sample():
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Test model on random input data.
  input_shape = input_details[0]['shape']
  latent = (np.random.random_sample(input_shape) - 0.5) * 2.0
  input_data = np.array(latent, dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  result = interpreter.get_tensor(output_details[0]['index'])
  result = np.reshape(result, [256,256,3])
  result = (result + 1.0) * 127.5
  result = pygame.surfarray.make_surface(result)
  result = pygame.transform.rotate(result, -90)
  return result

pygame.init()
display = pygame.display.set_mode((300, 300))
surface = sample()

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                surface = sample()
    display.blit(surface, (22, 22))
    pygame.display.update()
pygame.quit()
