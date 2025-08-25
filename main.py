import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

# Download/load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


print(f"before: Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"before: Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")


# lower the dataset size for quicker training during testing
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:2000], y_test[:2000]

print(f"after: Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"after: Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        # Multi-head attention
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class veaEncoderDecoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(veaEncoderDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder with CNN + Transformer
        self.cnn_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),  # 16x16
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),  # 8x8
            
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),  # 4x4
            
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
        ])
        
        # Transformer layers
        self.d_model = 512
        self.num_transformer_layers = 2
        self.transformer_blocks = [
            TransformerBlock(d_model=self.d_model, num_heads=8, dff=1024, dropout_rate=0.1)
            for _ in range(self.num_transformer_layers)
        ]
        
        # Final encoder layers
        self.encoder_final = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),  # Pool the spatial dimensions
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(latent_dim * 2),  # mean and logvar
        ])

        # Decoder with Transformer + CNN
        self.decoder_start = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4*4*512, activation='relu'),
            tf.keras.layers.Reshape((4, 4, 512)),
        ])
        
        # Transformer blocks for decoder
        self.decoder_transformer_blocks = [
            TransformerBlock(d_model=self.d_model, num_heads=8, dff=1024, dropout_rate=0.1)
            for _ in range(self.num_transformer_layers)
        ]
        
        # CNN Decoder
        self.cnn_decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same'),  # 8x8
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            
            tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'),  # 16x16
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            
            tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),  # 32x32
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'),
        ])

        # Build the model
        self._build_model()

    def _build_model(self):
        """Build the model to initialize layers"""
        dummy_input = tf.zeros((1, 32, 32, 3))
        # Build encoder path
        mean, logvar = self.encode(dummy_input, training=False)
        z = self.reparameterize(mean, logvar)
        # Build decoder path
        reconstructed = self.decode(z, training=False)
        # Build full model
        _ = self.call(dummy_input, training=False)

    def call(self, x, training=False):
        mean, logvar = self.encode(x, training=training)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z, training=training)
        return reconstructed

    def train_step(self, x):
        if isinstance(x, tuple):
            x = x[0]
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(x, training=True)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z, training=True)
            
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
        return self.decode(eps, training=False)
    
    def encode(self, x, training=False):
        # CNN feature extraction
        features = self.cnn_encoder(x, training=training)
        batch_size = tf.shape(features)[0]
        
        # Reshape for transformer: (batch, seq_len, d_model)
        features_reshaped = tf.reshape(features, (batch_size, -1, self.d_model))
        
        # Apply transformer blocks
        transformer_output = features_reshaped
        for transformer_block in self.transformer_blocks:
            transformer_output = transformer_block(transformer_output, training=training)
        
        # Reshape back to spatial format
        spatial_features = tf.reshape(transformer_output, tf.shape(features))
        
        # Final encoding
        mean_logvar = self.encoder_final(spatial_features, training=training)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean
    
    def decode(self, z, training=False):
        # Start decoding
        features = self.decoder_start(z, training=training)
        batch_size = tf.shape(features)[0]
        
        # Reshape for transformer: (batch, seq_len, d_model)
        features_reshaped = tf.reshape(features, (batch_size, -1, self.d_model))
        
        # Apply transformer blocks
        transformer_output = features_reshaped
        for transformer_block in self.decoder_transformer_blocks:
            transformer_output = transformer_block(transformer_output, training=training)
        
        # Reshape back to spatial format
        spatial_features = tf.reshape(transformer_output, tf.shape(features))
        
        # CNN decoding
        return self.cnn_decoder(spatial_features, training=training)


# Use larger latent dimension for the bigger model
latent_dim = 128  # Increased from 64
model = veaEncoderDecoder(latent_dim)

# Build the model explicitly with actual data shape
model.build(input_shape=(None, 32, 32, 3))

model.optimizer = tf.keras.optimizers.Adam(1e-4)

# Print model summary
print("Model created successfully!")
print(f"Latent dimension: {latent_dim}")
print(f"Number of transformer layers: {model.num_transformer_layers}")

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
epochs = 100
batch_size = 128
best_loss = float('inf')
patience = 20
wait = 0

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Build model with actual training data to ensure all weights are initialized
print("Building model with training data...")
sample_batch = next(iter(train_dataset))
_ = model(sample_batch[:1], training=False)
print("Model built successfully!")

print("Starting training...")
for epoch in range(epochs):
    epoch_loss = 0
    epoch_reconstruction_loss = 0
    epoch_kl_loss = 0
    num_batches = 0
    
    # Reset dataset for each epoch to avoid iterator issues
    epoch_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    for batch in epoch_dataset:
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
    
    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        wait = 0
        model.save_weights('best_vae_weights.weights.h5')
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print("Training completed!")

# Final visualization
print("\nFinal reconstructions:")
plot_reconstructions(model, x_test, 10)

print("\nGenerated samples:")
plot_generated_samples(model, 10)

