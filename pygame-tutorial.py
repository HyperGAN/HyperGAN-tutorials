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
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  result = interpreter.get_tensor(output_details[0]['index'])
  result = np.reshape(result, [128,128,3])
  return result

print(np.shape(sample()))

pygame.init()
display = pygame.display.set_mode((350, 350))
surf = pygame.surfarray.make_surface(sample())

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                surf = pygame.surfarray.make_surface(sample())
    display.blit(surf, (0, 0))
    pygame.display.update()
pygame.quit()
