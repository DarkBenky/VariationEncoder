import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

# Download/load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

class veaEncoderDecoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(veaEncoderDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim),  # mean and logvar
        ])

        self.encoder.summary()
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(4*4*128, activation='relu'),
            tf.keras.layers.Reshape((4, 4, 128)),
            tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'),
        ])

        self.decoder.summary()

    def train_step(self, x):
        if isinstance(x, tuple):
            x = x[0]
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z)
            
            # Compute reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x, x_logit)
            )
            reconstruction_loss *= 32 * 32 * 3
            
            # Compute KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                logvar - tf.square(mean) - tf.exp(logvar) + 1
            )
            
            total_loss = reconstruction_loss + kl_loss
            
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps)
    
    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean
    
    def decode(self, z):
        return self.decoder(z)


latent_dim = 64
model = veaEncoderDecoder(latent_dim)
model.optimizer = tf.keras.optimizers.Adam(1e-4)

def plot_reconstructions(model, test_images, num_images=10):
    """Plot original and reconstructed images"""
    test_sample = test_images[:num_images]
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    reconstructed = model.decode(z)
    
    fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
    for i in range(num_images):
        # Original images
        axes[0, i].imshow(test_sample[i])
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        # Reconstructed images
        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.show()

def plot_generated_samples(model, num_samples=10):
    """Generate and plot new samples from the VAE"""
    generated = model.sample(tf.random.normal(shape=(num_samples, model.latent_dim)))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
    for i in range(num_samples):
        axes[i].imshow(generated[i])
        axes[i].axis('off')
        axes[i].set_title('Generated')
    
    plt.tight_layout()
    plt.show()

# Training loop
epochs = 10
batch_size = 128

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(len(x_train)).batch(batch_size)

print("Starting training...")
for epoch in range(epochs):
    epoch_loss = 0
    epoch_reconstruction_loss = 0
    epoch_kl_loss = 0
    num_batches = 0
    
    for batch in train_dataset:
        losses = model.train_step(batch)
        epoch_loss += losses["loss"]
        epoch_reconstruction_loss += losses["reconstruction_loss"]
        epoch_kl_loss += losses["kl_loss"]
        num_batches += 1
    
    # Average losses
    epoch_loss /= num_batches
    epoch_reconstruction_loss /= num_batches
    epoch_kl_loss /= num_batches
    
    print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}, Reconstruction: {epoch_reconstruction_loss:.4f}, KL: {epoch_kl_loss:.4f}')
    if epoch % 10 == 0 and epoch != 0:
        # Show reconstructions every 10 epochs
        plot_reconstructions(model, x_test, 8)

print("Training completed!")

# Final visualization
print("\nFinal reconstructions:")
plot_reconstructions(model, x_test, 10)

print("\nGenerated samples:")
plot_generated_samples(model, 10)

