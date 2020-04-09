# PyTorch-ADDA
Editing corenel's framework for Adversarial Discriminative Domain Adaptation to work with x-ray images. Utilizing CheXpert's and NIH's xray databases.

## Environment
- Python 3.6
- PyTorch 0.2.0

## Usage

Run the following command:

```shell
python3 main.py
```

## Network

Uses three networks, ResNet18 encoder, classifier, and discrimnator

ResNet18 encoder and classifier comes from arnaghosh's implementation
https://github.com/arnaghosh/Auto-Encoder/blob/master/resnet.py

- Discriminator

  ```
  Discriminator (
    (layer): Sequential (
      (0): Linear (1000 -> 500)
      (1): ReLU ()
      (2): Linear (500 -> 500)
      (3): ReLU ()
      (4): Linear (500 -> 2)
      (5): LogSoftmax ()
    )
  )
  ```

## Result

tbd