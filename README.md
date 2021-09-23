# ImageNet Classification


## Model 
  1. DenseNet 169
  
## Model Specification
  1. Total layer size : 34
  2. Input image size : 224
  4. Input kernal size : 7
  5. Layer size list : [6, 12, 32, 32]
  6. Growth Rate : 24
  7. Kernal size : 3
  8. Class size : 1000

## Data Augmentation
  1. Train
      > 1) Resize (size : 256)
      > 2) RandomHorizontalFlip
      > 3) RandomCrop(size : 224)
      > 4) Normalize
      > 5) CutMix

  2. Test
      > 1) Resize (size : 225)
      > 2) Normalize

## Training 
  1. Optimizer : SGD (momentum = 0.9)
  2. Scheudler : StepLR
      * Init learning rate : 1e-2
      * Step Size : 20
      * Gamma = 0.1
  3. Epochs : 100
  4. Batch size : 256

## Data 
  1. ImageNet
      1. Data Size : 700000
      2. Class Size : 1000

## Reference
  1. Densenet : https://arxiv.org/pdf/1608.06993.pdf
  2. CutMix : https://arxiv.org/pdf/1905.04899.pdf

