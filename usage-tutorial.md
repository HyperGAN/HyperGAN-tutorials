image
Train a custom cifar generator using hypergan

## Step 1: Acquire the dataset

There are a lot of publicly available datasets and you can create your own. [www.academictorrents.com] is an amazing p2p way to find datasets.

We will use a small dataset of cifar.  From [https://pjreddie.com/projects/cifar-10-dataset-mirror/]

`wget http://pjreddie.com/media/files/cifar.tgz`

`tar xvzf cifar.tgz`

## Step 2: Install HyperGAN

To train with HyperGAN, you should have a GPU.  CPU training is extremely slow.

HyperGAN depends on tensorflow which depends on cuda and cudnn.

`pip3 install hypergan --upgrade`

Detailed installation instructions can be found [https://github.com/HyperGAN/HyperGAN#install]

## Step 3: Train with the 'default' config

`hypergan train cifar --size 32x32x3`

If everything is set up correctly, your GAN should start training after a small load time.

## Step 4: Try other configurations to increase quality

Configurations define the GAN network topology, training regimine, as well as hyperparameters.  This means different configurations can lead to very different trained generators.

HyperGAN includes many precreated configurations.  We can list all of them with:

`hypergan new -l .`

There are a lot, lets choose wgan-gp - a well known configuration that combines gradient penalty with wasserstein loss.

`hypergan new -c wgan-gp myconfig`

This will create a wgan-gp.json file.  You can edit that file if you'd like to see how the network is configured.

We can train with our new configuration with the `-c` argument.

`hypergan train [dataset] --format png -c myconfig`


Note you can train multiple HyperGAN models concurrently.

To restrict the gpu being used add `CUDA_VISIBLE_DEVICES=0`
Use nvidia-smi to figure out which cards number should be set.

## Step 5: Build your creation

`hypergan build -c`

This will create a `build/default.tflite` file.  You can build another configuration with `-c myconfig`. This is the separated generator for your model.  Everything has been condensed and optimized.

`default` trained in 150MB of gpu ram but condenses to 1.6 MB.

You can now use this tflite model in your mobile apps, games and websites.

## Step 6: Share your creation

HyperGAN is more than a tool to build GANs.  It's a community driven project where you can share your creations.

Join the community discord and post to the #showcase room.
