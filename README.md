# Unsupervised Image-to-Image Translation Network

## Inputs
512x512x3 random crops from a 1280x720 image, normalised to [-1, 1].
Dataset from BDD100k. 
## Encoder
* Conv 7x7x64, stride 1, InstNorm, ReLU
* Conv 3x3x128, stride 2, InstNorm, ReLU
* Conv 3x3x256, stride 2, InstNorm, ReLU
* ResBlock 3x3x256, InstNorm, ReLU
* ResBlock 3x3x256, InstNorm, ReLU
* ResBlock 3x3x256, InstNorm, ReLU
* ResBlock 3x3x256, InstNorm, ReLU

One ResBlock is: Conv - Norm - Activ - Conv - Norm - __Add__ - Activ.

No norm or activation for last layer, but add bias.

## Decoder
* ResBlock 3x3x256, InstNorm, ReLU
* ResBlock 3x3x256, InstNorm, ReLU
* ResBlock 3x3x256, InstNorm, ReLU
* ResBlock 3x3x256, InstNorm, ReLU
* Upsample factor 2
* Conv 5x5x128, InstNorm, ReLU (but LayerNorm might be better)
* Upsample factor 2
* Conv 5x5x64, InstNorm, ReLU (but LayerNorm might be better)
* Conv 7x7x3, Tanh

Don't forget bias in the last layer.

## Discriminator
* Conv 3x3x64, stride 2, LReLU
* Conv 3x3x128, stride 2, LReLU
* Conv 3x3x256, stride 2, LReLU
* Conv 3x3x512, stride 2, LReLU
* Conv 1x1x1, stride 1

Repeat this discri architecture 3 times (AvgPool 3x3, stride 2 
applied in-between).

## Training
* Update discriminators weights with generators fixed.
* Update generators weights with discriminator fixed.

