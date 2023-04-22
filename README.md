# Conditional GAN trained on MNIST dataset

## 2500 generated handwritten digits
![2500 generated handwritten digits](./docs/generated_images.png)

## Training
![Training history](./docs/training_history.png)

## Load trained model

```python
g_opt = Adam(learning_rate=0.0001, beta_1=0.5)
d_opt = Adam(learning_rate=0.00001, beta_1=0.5)

g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

generator = build_generator()
discriminator = build_discriminator()

gan = GAN(generator, discriminator)
gan.compile(g_opt, d_opt, g_loss, d_loss)

gan.load_weights('./checkpoints/my_checkpoint')
```
