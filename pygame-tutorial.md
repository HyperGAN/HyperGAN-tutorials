
Train a GAN and use the generator with pygame.

### A prepackaged generator

You should first use the prepackaged generator here:

https://hypergan.s3-us-west-1.amazonaws.com/0.10/tutorial1.tflite

```
wget https://hypergan.s3-us-west-1.amazonaws.com/0.10/tutorial1.tflite
```

### Load the tflite model

```
import numpy as np
import tensorflow as tf

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
  return interpreter.get_tensor(output_details[0]['index'])
```
**From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/inference.md#load-and-run-a-model-in-python **

### Render the model to a bitmap

```python
import pygame
pygame.init()
display = pygame.display.set_mode((350, 350))
surf = pygame.surfarray.make_surface(sample())

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    display.blit(surf, (0, 0))
    pygame.display.update()
pygame.quit()
```

### Randomize the latent variable

```
if event.type == pygame.KEYDOWN:
    if event.key == pygame.K_SPACE:
      surf = pygame.surfarray.make_surface(sample())
```

### Putting it all together

See (pygame.py)[pygame.py]

## Create your own model

If you want to train a model from scratch, you will need:

* a HyperGAN training environment
* a GPU
* a dataset directory of images to train against

### Train your model

```
hypergan train [dataset]
```

This will take several hours.  A view will display the training progress.

You will need to save and quit the model when you are satisfied with the results.


### Build the model

```
hypergan build
```

This will generate a `tflite` file in your build directory.

### Fine tune your results

There are many differing configurations you can use to train your GAN and each decision will effect the final output.

You can see all the prepacked configurations with:

```
hypergan new . -l
```

More information and help can be found in the discord.

