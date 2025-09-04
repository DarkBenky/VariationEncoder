import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Input
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import wandb
import io

# check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

# Configure GPU memory growth to avoid OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU memory configuration error: {e}")

# Initialize wandb
wandb.init(
    project="VAE",
    config={
        "latent_dim": 256,
        "batch_size": 64,
        "epochs": 25,
        "learning_rate_vae": 0.0002,
        "learning_rate_classifier": 0.001,
        "learning_rate_inverse": 0.001,
        "learning_rate_gan": 0.0001,
        "architecture": "enhanced_inverse_classifier",
        "gan_epochs": 10,
        "gan_generator_lr": 0.0002,
        "gan_discriminator_lr": 0.0001
    }
)
# 1. Load the dataset and metadata
ds_train, ds_info = tfds.load(
    "celeb_a",
    split="train",
    shuffle_files=True,
    with_info=True,
    as_supervised=False
)

print(ds_info)  
print(ds_info.features)

def buildEncoder(latent_dim):
    encoder_input = Input(shape=(256, 256, 3), name="encoder_input")
    x = Conv2D(32, 4, strides=2, padding="same")(encoder_input)  # 128x128
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Conv2D(64, 4, strides=2, padding="same")(x)  # 64x64
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Conv2D(128, 4, strides=2, padding="same")(x)  # 32x32
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Conv2D(256, 4, strides=2, padding="same")(x)  # 16x16
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Conv2D(512, 4, strides=2, padding="same")(x)  # 8x8
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    
    # VAE latent space with mean and log variance
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
    return encoder

def buildClassifier(latent_dim):
    classifier_input = Input(shape=(latent_dim,), name="classifier_input")
    x = Dense(256, activation="relu")(classifier_input)
    x = Dense(512, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    classifier_output = Dense(40, activation="sigmoid", name="classifier_output")(x)
    classifier = Model(classifier_input, classifier_output, name="classifier")
    return classifier

def train_classifier(data, encoder, classifier, epochs=10, batch_size=32, latent_dim=128):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(x, training=False)
            y_pred = classifier(z, training=True)
            loss = bce_loss_fn(y, y_pred)
            
            # Add L2 regularization
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in classifier.trainable_weights])
            total_loss = loss + 0.0001 * l2_loss
        
        grads = tape.gradient(total_loss, classifier.trainable_weights)
        optimizer.apply_gradients(zip(grads, classifier.trainable_weights))
        return total_loss, loss

    for epoch in range(epochs):
        print(f"Classifier Epoch {epoch + 1}/{epochs}")
        epoch_total_loss = 0
        epoch_bce_loss = 0
        step_count = 0
        
        for step, batch in enumerate(data):
            images = tf.cast(batch['image'], tf.float32) / 255.0
            
            # Convert nested attributes dict to tensor array
            attr_list = []
            for attr_name in sorted(batch['attributes'].keys()):  # Sort for consistent order
                attr_list.append(tf.cast(batch['attributes'][attr_name], tf.float32))
            attributes = tf.stack(attr_list, axis=1)  # Shape: (batch_size, 40)
            
            total_loss, loss = train_step(images, attributes)
            epoch_total_loss += total_loss.numpy()
            epoch_bce_loss += loss.numpy()
            step_count += 1
            
            if step % 100 == 0:
                print(f"Step {step}, Total Loss: {total_loss.numpy():.6f}, BCE Loss: {loss.numpy():.6f}")
                
                # Log to wandb
                wandb.log({
                    "classifier/step_total_loss": total_loss.numpy(),
                    "classifier/step_bce_loss": loss.numpy(),
                    "classifier/epoch": epoch + 1,
                    "classifier/step": step
                })
        
        # Log epoch averages
        avg_total_loss = epoch_total_loss / step_count
        avg_bce_loss = epoch_bce_loss / step_count
        
        wandb.log({
            "classifier/epoch_avg_total_loss": avg_total_loss,
            "classifier/epoch_avg_bce_loss": avg_bce_loss,
            "classifier/epoch_completed": epoch + 1
        })
        
        # Save model checkpoints
        classifier.save_weights(f"classifier.weights.h5")
    print("Classifier training completed.")

def buildDecoder(latent_dim):
    decoder_input = Input(shape=(latent_dim,), name="decoder_input")
    
    # Enhanced initial projection with more capacity
    x = Dense(16 * 16 * 1024, activation="relu")(decoder_input)  # Larger initial projection
    x = tf.keras.layers.BatchNormalization()(x)
    x = Reshape((16, 16, 1024))(x)
    
    # First upsampling block with attention
    x = Conv2DTranspose(512, 4, strides=2, padding="same")(x)  # 32x32
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Self-attention layer for better feature correlation
    attention_1 = tf.keras.layers.MultiHeadAttention(
        num_heads=8, key_dim=64, name="attention_32x32"
    )
    # Flatten for attention, then reshape back
    x_flat = tf.keras.layers.Reshape((32*32, 512))(x)
    x_attended = attention_1(x_flat, x_flat)
    x = tf.keras.layers.Reshape((32, 32, 512))(x_attended)
    
    # Second upsampling block
    x = Conv2DTranspose(256, 4, strides=2, padding="same")(x)  # 64x64
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.1)(x)  # Light dropout for regularization
    
    # Third upsampling block
    x = Conv2DTranspose(128, 4, strides=2, padding="same")(x)  # 128x128
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Fourth upsampling block with residual connection
    residual_128 = Conv2D(64, 1, padding="same")(x)  # Match channels for residual
    x = Conv2DTranspose(64, 4, strides=2, padding="same")(x)   # 256x256
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Add residual connection (upsample residual to match dimensions)
    residual_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2))(residual_128)
    x = tf.keras.layers.Add()([x, residual_upsampled])
    
    # Additional refinement layers
    x = Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(16, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Final output layer with tanh activation scaled to [0,1]
    decoder_output = Conv2D(3, 3, activation="sigmoid", padding="same", name="decoder_output")(x)
    
    decoder = Model(decoder_input, decoder_output, name="enhanced_decoder")
    return decoder

def vae_loss(x, x_decoded_mean, z_mean, z_log_var):
    # Reconstruction loss
    reconstruction_loss = tf.keras.losses.binary_crossentropy(
        tf.keras.backend.flatten(x),
        tf.keras.backend.flatten(x_decoded_mean)
    )
    reconstruction_loss *= 256 * 256
    
    # KL divergence loss
    kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    # Perceptual loss using VGG16
    vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    vgg.trainable = False
    
    x_vgg = vgg(x * 255)
    x_decoded_vgg = vgg(x_decoded_mean * 255)
    perceptual_loss = tf.keras.losses.mse(x_vgg, x_decoded_vgg)
    
    return tf.keras.backend.mean(reconstruction_loss + 0.1 * kl_loss + 0.01 * perceptual_loss)

def train(data, encoder, decoder, epochs=10, batch_size=32, latent_dim=128):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    best_loss = float('inf')
    
    @tf.function
    def train_step(x, step):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(x, training=True)
            x_recon = decoder(z, training=True)
            
            # Calculate VAE loss components
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.reshape(x, [-1, 256*256*3]),
                    tf.reshape(x_recon, [-1, 256*256*3])
                )
            ) * 256 * 256 * 3
            
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            # Progressive KL weight
            kl_weight = tf.minimum(1.0, tf.cast(step, tf.float32) / 10000.0)
            
            total_loss = reconstruction_loss + kl_weight * kl_loss
        
        grads = tape.gradient(total_loss, encoder.trainable_weights + decoder.trainable_weights)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]  # Gradient clipping
        optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
        
        return total_loss, reconstruction_loss, kl_loss

    for epoch in range(epochs):
        print(f"VAE Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        step_count = 0
        
        for step, batch in enumerate(data):
            images = tf.cast(batch['image'], tf.float32) / 255.0
            
            # Data augmentation
            if tf.random.uniform([]) > 0.5:
                images = tf.image.flip_left_right(images)
            images = tf.image.random_brightness(images, 0.1)
            images = tf.image.random_contrast(images, 0.9, 1.1)
            
            loss, recon_loss, kl_loss = train_step(images, global_step)
            epoch_loss += loss
            epoch_recon_loss += recon_loss
            epoch_kl_loss += kl_loss
            step_count += 1
            global_step.assign_add(1)
            
            if step % 100 == 0:
                print(f"Step {step}, Total: {loss.numpy():.4f}, Recon: {recon_loss.numpy():.4f}, KL: {kl_loss.numpy():.4f}")
                
                # Log to wandb
                wandb.log({
                    "vae/step_total_loss": loss.numpy(),
                    "vae/step_recon_loss": recon_loss.numpy(),
                    "vae/step_kl_loss": kl_loss.numpy(),
                    "vae/epoch": epoch + 1,
                    "vae/step": step,
                    "vae/global_step": global_step.numpy()
                })
        
        avg_loss = epoch_loss / step_count
        avg_recon_loss = epoch_recon_loss / step_count
        avg_kl_loss = epoch_kl_loss / step_count
        
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        
        # Log epoch averages
        wandb.log({
            "vae/epoch_avg_total_loss": avg_loss.numpy(),
            "vae/epoch_avg_recon_loss": avg_recon_loss.numpy(),
            "vae/epoch_avg_kl_loss": avg_kl_loss.numpy(),
            "vae/epoch_completed": epoch + 1
        })
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  ✓ New best model saved! Loss: {best_loss:.6f}")
            encoder.save_weights("encoder.weights.h5")
            decoder.save_weights("decoder.weights.h5")
            wandb.log({
                "vae/new_best_loss": best_loss.numpy(),
                "vae/best_epoch": epoch + 1
            })
    
    print("Training completed.")

def buildInverseClassifier(latent_dim):
    """Ultra-enhanced network with attention, residuals, and advanced regularization"""
    inverse_input = Input(shape=(40,), name="inverse_input")  # 40 CelebA attributes
    
    # Enhanced input embedding with feature importance weighting
    x = Dense(1024, activation="relu")(inverse_input)
    x = tf.keras.layers.LayerNormalization()(x)  # Layer norm instead of batch norm for stability
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Feature importance branch
    importance_weights = Dense(1024, activation="sigmoid", name="feature_importance")(x)
    x = tf.keras.layers.Multiply()([x, importance_weights])
    
    # First enhanced residual block with attention
    residual1 = x
    x = Dense(2048, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Multi-head self-attention
    x_reshaped = tf.keras.layers.Reshape((1, 2048))(x)
    attention_1 = tf.keras.layers.MultiHeadAttention(
        num_heads=16, key_dim=128, name="attention_1"
    )
    x_attended = attention_1(x_reshaped, x_reshaped)
    x = tf.keras.layers.Reshape((2048,))(x_attended)
    
    x = Dense(1024, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Add()([x, residual1])  # Residual connection
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Second enhanced residual block
    residual2 = x
    x = Dense(2048, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = Dense(1024, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Add()([x, residual2])
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Third enhanced residual block
    residual3 = x
    x = Dense(2048, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = Dense(1024, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Add()([x, residual3])
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Fourth residual block with cross-attention to input
    residual4 = x
    x = Dense(2048, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Cross-attention between current features and original input
    input_expanded = Dense(2048, activation="relu")(inverse_input)
    x_reshaped = tf.keras.layers.Reshape((1, 2048))(x)
    input_reshaped = tf.keras.layers.Reshape((1, 2048))(input_expanded)
    
    cross_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=16, key_dim=128, name="cross_attention"
    )
    x_cross_attended = cross_attention(x_reshaped, input_reshaped)
    x = tf.keras.layers.Reshape((2048,))(x_cross_attended)
    
    x = Dense(1024, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Add()([x, residual4])
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Additional deep processing layers
    x = Dense(1536, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = Dense(1024, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = Dense(768, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    
    # Pre-output processing with gating mechanism
    gate = Dense(512, activation="sigmoid")(x)
    x = Dense(512, activation="relu")(x)
    x = tf.keras.layers.Multiply()([x, gate])  # Gated activation
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Final projection to latent space with residual connection
    pre_output = Dense(latent_dim)(x)
    
    # Additional refinement branch
    refinement = Dense(latent_dim // 2, activation="relu")(x)
    refinement = Dense(latent_dim)(refinement)
    
    # Combine main output with refinement
    inverse_output = tf.keras.layers.Add()([pre_output, refinement])
    inverse_output = tf.keras.layers.LayerNormalization(name="inverse_output")(inverse_output)
    
    inverse_classifier = Model(inverse_input, inverse_output, name="ultra_enhanced_inverse_classifier")
    return inverse_classifier

def train_inverse_classifier(data, encoder, inverse_classifier, epochs=10, batch_size=32, latent_dim=128):
    # Enhanced learning rate scheduling with warm-up - Fixed
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.001,
        decay_steps=epochs * 1000,
        end_learning_rate=0.0001,
        power=0.9
    )
    
    # Enhanced optimizer with gradient centralization
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, 
        beta_1=0.9, 
        beta_2=0.999,
        weight_decay=0.0001,  # L2 regularization built into optimizer
        epsilon=1e-7
    )
    
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    cosine_loss_fn = tf.keras.losses.CosineSimilarity()
    huber_loss_fn = tf.keras.losses.Huber(delta=1.0)  # More robust to outliers
    
    # Track best loss for saving best model
    best_loss = float('inf')
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            # Get encoder outputs (use z_mean for stability)
            z_mean, z_log_var, z_sample = encoder(x, training=False)
            
            # Get inverse classifier prediction
            z_pred = inverse_classifier(y, training=True)
            
            # Enhanced loss combination
            mse_loss = mse_loss_fn(z_mean, z_pred)
            cosine_loss = 1.0 - cosine_loss_fn(z_mean, z_pred)
            huber_loss = huber_loss_fn(z_mean, z_pred)
            
            # Adaptive loss weighting based on training progress
            mse_weight = 0.5
            cosine_weight = 0.3
            huber_weight = 0.2
            
            reconstruction_loss = (mse_weight * mse_loss + 
                                 cosine_weight * cosine_loss + 
                                 huber_weight * huber_loss)
            
            # Enhanced regularization with different penalties
            l2_loss = 0
            for layer in inverse_classifier.layers:
                if hasattr(layer, 'kernel'):
                    if hasattr(layer, 'units') and layer.units >= 1024:
                        l2_loss += 0.00005 * tf.nn.l2_loss(layer.kernel)
                    elif hasattr(layer, 'units') and layer.units >= 512:
                        l2_loss += 0.0001 * tf.nn.l2_loss(layer.kernel)
                    else:
                        l2_loss += 0.00005 * tf.nn.l2_loss(layer.kernel)
            
            # Total loss
            total_loss = reconstruction_loss + l2_loss
            
        # Enhanced gradient computation
        grads = tape.gradient(total_loss, inverse_classifier.trainable_weights)
        
        # Gradient clipping and normalization
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        
        # Apply gradients
        optimizer.apply_gradients(zip(grads, inverse_classifier.trainable_weights))
        
        return total_loss, mse_loss, cosine_loss, huber_loss

    # Training metrics tracking
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    mse_metric = tf.keras.metrics.Mean(name='mse_loss')
    cosine_metric = tf.keras.metrics.Mean(name='cosine_loss')
    huber_metric = tf.keras.metrics.Mean(name='huber_loss')

    for epoch in range(epochs):
        print(f"Ultra Enhanced Inverse Classifier Epoch {epoch + 1}/{epochs}")
        
        # Reset metrics
        train_loss_metric.reset_state()
        mse_metric.reset_state()
        cosine_metric.reset_state()
        huber_metric.reset_state()
        
        for step, batch in enumerate(data):
            images = tf.cast(batch['image'], tf.float32) / 255.0
            
            # Convert nested attributes dict to tensor array
            attr_list = []
            for attr_name in sorted(batch['attributes'].keys()):
                attr_list.append(tf.cast(batch['attributes'][attr_name], tf.float32))
            attributes = tf.stack(attr_list, axis=1)
            
            # Enhanced data augmentation for attributes
            noise_std = 0.03 * (1.0 - epoch / epochs)  # Reduce noise over time
            noise = tf.random.normal(tf.shape(attributes), stddev=noise_std)
            attributes_smoothed = tf.clip_by_value(attributes + noise, 0.0, 1.0)
            
            total_loss, mse_loss, cosine_loss, huber_loss = train_step(images, attributes_smoothed)
            
            # Update metrics
            train_loss_metric.update_state(total_loss)
            mse_metric.update_state(mse_loss)
            cosine_metric.update_state(cosine_loss)
            huber_metric.update_state(huber_loss)
            
            if step % 100 == 0:
                print(f"Step {step}, Total: {total_loss.numpy():.6f}, "
                      f"MSE: {mse_loss.numpy():.6f}, Cosine: {cosine_loss.numpy():.6f}, "
                      f"Huber: {huber_loss.numpy():.6f}, LR: {optimizer.learning_rate.numpy():.6f}")
                
                # Log to wandb
                wandb.log({
                    "inverse_classifier/step_total_loss": total_loss.numpy(),
                    "inverse_classifier/step_mse_loss": mse_loss.numpy(),
                    "inverse_classifier/step_cosine_loss": cosine_loss.numpy(),
                    "inverse_classifier/step_huber_loss": huber_loss.numpy(),
                    "inverse_classifier/learning_rate": optimizer.learning_rate.numpy(),
                    "inverse_classifier/epoch": epoch + 1,
                    "inverse_classifier/step": step
                })
        
        # Calculate epoch average loss
        epoch_avg_loss = train_loss_metric.result().numpy()
        
        # Print epoch summary
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Total Loss: {epoch_avg_loss:.6f}")
        print(f"  Average MSE Loss: {mse_metric.result():.6f}")
        print(f"  Average Cosine Loss: {cosine_metric.result():.6f}")
        print(f"  Average Huber Loss: {huber_metric.result():.6f}")
        
        # Log epoch metrics to wandb
        wandb.log({
            "inverse_classifier/epoch_avg_total_loss": epoch_avg_loss,
            "inverse_classifier/epoch_avg_mse_loss": mse_metric.result().numpy(),
            "inverse_classifier/epoch_avg_cosine_loss": cosine_metric.result().numpy(),
            "inverse_classifier/epoch_avg_huber_loss": huber_metric.result().numpy(),
            "inverse_classifier/epoch_completed": epoch + 1,
            "inverse_classifier/best_loss": best_loss
        })
        
        # Save only if this is the best model so far
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            inverse_classifier.save_weights("inverse_classifier.weights.h5")
            print(f"  ✓ New best model saved! Loss: {best_loss:.6f}")
            
            # Log best model update to wandb
            wandb.log({
                "inverse_classifier/new_best_loss": best_loss,
                "inverse_classifier/best_epoch": epoch + 1
            })
        else:
            print(f"  Current best loss: {best_loss:.6f}")
    
    print(f"Ultra enhanced inverse classifier training completed. Best loss: {best_loss:.6f}")

def generate_image_from_features(feature_list, inverse_classifier, decoder, attribute_names=None):
    features = tf.constant([feature_list], dtype=tf.float32)
    latent_code = inverse_classifier(features, training=False)
    
    # Add slight noise for variation
    noise = tf.random.normal(tf.shape(latent_code), stddev=0.1)
    latent_code = latent_code + noise
    
    generated_image = decoder(latent_code, training=False)
    return generated_image[0].numpy()

def format_attributes(attributes_dict, ds_info, max_attrs=5):
    """Format CelebA attributes into readable descriptions"""
    active_attrs = []
    
    for attr_name, attr_value in attributes_dict.items():
        if attr_value == 1:  # attribute is present
            # Clean up attribute names for display
            clean_name = attr_name.replace('_', ' ').title()
            active_attrs.append(clean_name)
    
    # Limit number of attributes shown
    if len(active_attrs) > max_attrs:
        shown_attrs = active_attrs[:max_attrs]
        description = ", ".join(shown_attrs) + f" (+{len(active_attrs)-max_attrs} more)"
    else:
        description = ", ".join(active_attrs) if active_attrs else "No notable attributes"
    
    return description

def display_reconstructions(encoder, decoder, data, num_images=5):
    sample_batch = next(iter(data))
    images = tf.cast(sample_batch['image'], tf.float32) / 255.0
    
    # Debug: print image stats
    print(f"Display - Image shape: {images.shape}")
    print(f"Display - Image min/max: {tf.reduce_min(images).numpy():.4f} / {tf.reduce_max(images).numpy():.4f}")
    
    z_mean, z_log_var, z = encoder(images[:num_images])
    reconstructions = decoder(z)

    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        # Get attributes for this image - handle nested dict structure
        attrs_dict = {}
        for attr_name in sample_batch['attributes'].keys():
            attrs_dict[attr_name] = sample_batch['attributes'][attr_name][i].numpy()
        description = format_attributes(attrs_dict, ds_info)
        
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(np.clip(images[i].numpy(), 0, 1))
        plt.axis("off")
        if i == 0:
            ax.set_title("Originals", fontsize=12, fontweight='bold')
        
        # Reconstructed
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(np.clip(reconstructions[i].numpy(), 0, 1))
        plt.axis("off")
        if i == 0:
            ax.set_title("Reconstructions", fontsize=12, fontweight='bold')
            
        # Description
        ax = plt.subplot(3, num_images, i + 1 + 2*num_images)
        plt.text(0.5, 0.5, description, ha='center', va='center', wrap=True, 
                fontsize=8, transform=ax.transAxes)
        plt.axis("off")
        if i == 0:
            ax.set_title("Attributes", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def get_celeba_attribute_names():
    """Get the ordered list of CelebA attribute names"""
    return [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
        'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
        'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
        'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]

def create_feature_vector_from_descriptions(descriptions, attribute_names=None, intensities=None):
    if attribute_names is None:
        attribute_names = get_celeba_attribute_names()
    
    feature_vector = [0.0] * len(attribute_names)
    
    if intensities is None:
        intensities = [1.0] * len(descriptions)
    
    for desc, intensity in zip(descriptions, intensities):
        try:
            idx = attribute_names.index(desc)
            feature_vector[idx] = max(0.0, min(1.0, intensity))  # Clamp to 0-1
        except ValueError:
            print(f"Warning: '{desc}' not found in attribute names")
    
    return feature_vector

def log_generated_images_to_wandb(inverse_classifier, decoder, epoch_prefix=""):
    """Generate and log sample images to wandb"""
    try:
        # Generate different types of images
        test_cases = [
            (['Young', 'Smiling', 'Attractive'], [0.8, 0.9, 0.8], "Young_Smiling_Attractive"),
            (['Male', 'Mustache', 'Young'], [0.9, 0.7, 0.8], "Male_Mustache_Young"),
            (['Blond_Hair', 'Young', 'Smiling'], [0.8, 0.7, 0.6], "Blond_Smiling_Young"),
            (['Eyeglasses', 'Young', 'Male'], [0.9, 0.8, 0.9], "Glasses_Young_Male")
        ]
        
        images_to_log = {}
        
        for features, intensities, name in test_cases:
            feature_vector = create_feature_vector_from_descriptions(features, intensities=intensities)
            generated_image = generate_image_from_features(feature_vector, inverse_classifier, decoder)
            
            # Convert to PIL Image for wandb
            generated_image_clipped = np.clip(generated_image, 0, 1)
            images_to_log[f"{epoch_prefix}generated_{name}"] = wandb.Image(generated_image_clipped, caption=f"{name}: {', '.join(features)}")
        
        wandb.log(images_to_log)
        
    except Exception as e:
        print(f"Warning: Could not log images to wandb: {e}")


def buildRealFakeClassifier():
    """Enhanced real/fake classifier with attention, residual connections, and advanced regularization"""
    input_layer = Input(shape=(256, 256, 3), name="real_fake_input")
    
    # Initial normalization and preprocessing
    x = tf.keras.layers.BatchNormalization()(input_layer)
    
    # First conv block with residual connection preparation
    x = Conv2D(64, 4, strides=2, padding="same", kernel_initializer='he_normal')(x)  # 128x128
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # Second conv block with residual
    residual_64 = Conv2D(128, 1, strides=2, padding="same")(x)  # Match dimensions for residual
    x = Conv2D(128, 4, strides=2, padding="same", kernel_initializer='he_normal')(x)  # 64x64
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.Add()([x, residual_64])  # Residual connection
    x = tf.keras.layers.Dropout(0.15)(x)
    
    # Third conv block with attention
    x = Conv2D(256, 4, strides=2, padding="same", kernel_initializer='he_normal')(x)  # 32x32
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Simplified attention mechanism to reduce memory usage
    # Use channel attention instead of spatial attention
    attention_weights = tf.keras.layers.GlobalAveragePooling2D()(x)
    attention_weights = Dense(256 // 8, activation='relu')(attention_weights)
    attention_weights = Dense(256, activation='sigmoid')(attention_weights)
    attention_weights = tf.keras.layers.Reshape((1, 1, 256))(attention_weights)
    x = tf.keras.layers.Multiply()([x, attention_weights])
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Fourth conv block with residual
    residual_256 = Conv2D(512, 1, strides=2, padding="same")(x)
    x = Conv2D(512, 4, strides=2, padding="same", kernel_initializer='he_normal')(x)  # 16x16
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.Add()([x, residual_256])
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Fifth conv block with squeeze-and-excitation
    x = Conv2D(512, 4, strides=2, padding="same", kernel_initializer='he_normal')(x)  # 8x8
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Squeeze-and-Excitation block
    se_ratio = 16
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = Dense(512 // se_ratio, activation='relu')(se)
    se = Dense(512, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, 512))(se)
    x = tf.keras.layers.Multiply()([x, se])
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Additional conv block for more capacity
    x = Conv2D(1024, 3, strides=2, padding="same", kernel_initializer='he_normal')(x)  # 4x4
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Global pooling with both average and max
    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    gmp = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Concatenate()([gap, gmp])  # Combine both pooling methods
    
    # Enhanced dense layers with residual connections
    x = Dense(2048, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Residual dense block
    residual_dense = Dense(1024, kernel_initializer='he_normal')(x)
    x = Dense(1024, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.Add()([x, residual_dense])
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Feature importance weighting
    importance = Dense(1024, activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, importance])
    
    # Final classification layers
    x = Dense(512, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = Dense(256, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output with label smoothing preparation
    output = Dense(1, activation="sigmoid", name="real_fake_output")(x)
    
    model = Model(input_layer, output, name="enhanced_real_fake_classifier")
    return model

def train_real_fake_classifier(real_data, encoder, decoder, classifier, epochs=15, batch_size=32):
    """Enhanced training function for the real/fake classifier"""
    
    # Learning rate schedule with warmup
    def lr_schedule(epoch):
        if epoch < 2:
            return 0.0001 * (epoch + 1) / 2  # Warmup
        elif epoch < 10:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (10 - epoch))
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Enhanced loss with label smoothing
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.1,  # Prevents overconfident predictions
        from_logits=False
    )
    
    # Focal loss for handling class imbalance
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = alpha_t * tf.pow(1 - p_t, gamma) * bce_loss_fn(y_true, y_pred)
        return tf.reduce_mean(fl)
    
    # Metrics
    train_accuracy = tf.keras.metrics.BinaryAccuracy()
    train_precision = tf.keras.metrics.Precision()
    train_recall = tf.keras.metrics.Recall()
    
    @tf.function
    def train_step(real_images, fake_images):
        # Combine real and fake images
        batch_size = tf.shape(real_images)[0]
        images = tf.concat([real_images, fake_images], axis=0)
        
        # Create labels (1 for real, 0 for fake)
        real_labels = tf.ones((batch_size, 1), dtype=tf.float32)
        fake_labels = tf.zeros((batch_size, 1), dtype=tf.float32)
        labels = tf.concat([real_labels, fake_labels], axis=0)
        
        # Add noise to labels for robustness (flip small percentage)
        noise_factor = 0.05
        random_mask = tf.random.uniform(tf.shape(labels)) < noise_factor
        noisy_labels = tf.where(random_mask, 1.0 - labels, labels)
        
        with tf.GradientTape() as tape:
            predictions = classifier(images, training=True)
            
            # Combine focal loss and BCE loss
            bce_loss = bce_loss_fn(noisy_labels, predictions)
            focal_loss_val = focal_loss(noisy_labels, predictions)
            
            # L2 regularization
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in classifier.trainable_weights])
            
            total_loss = 0.7 * bce_loss + 0.3 * focal_loss_val + 0.0001 * l2_loss
        
        gradients = tape.gradient(total_loss, classifier.trainable_weights)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]  # Gradient clipping
        optimizer.apply_gradients(zip(gradients, classifier.trainable_weights))
        
        # Update metrics
        train_accuracy.update_state(noisy_labels, predictions)
        train_precision.update_state(noisy_labels, predictions)
        train_recall.update_state(noisy_labels, predictions)
        
        return total_loss, bce_loss, focal_loss_val
    
    print("Training Enhanced Real/Fake Classifier...")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Update learning rate
        current_lr = lr_schedule(epoch)
        optimizer.learning_rate.assign(current_lr)
        
        # Reset metrics
        train_accuracy.reset_state()
        train_precision.reset_state()
        train_recall.reset_state()
        
        epoch_loss = 0
        epoch_bce_loss = 0
        epoch_focal_loss = 0
        step_count = 0
        
        for step, batch in enumerate(real_data):
            real_images = tf.cast(batch['image'], tf.float32) / 255.0
            
            # Generate fake images
            batch_size = tf.shape(real_images)[0]
            random_z = tf.random.normal((batch_size, 256))
            fake_images = decoder(random_z, training=False)
            
            # Add data augmentation
            if tf.random.uniform([]) > 0.5:
                real_images = tf.image.flip_left_right(real_images)
                fake_images = tf.image.flip_left_right(fake_images)
            
            real_images = tf.image.random_brightness(real_images, 0.1)
            fake_images = tf.image.random_brightness(fake_images, 0.1)
            
            total_loss, bce_loss, focal_loss_val = train_step(real_images, fake_images)
            
            epoch_loss += total_loss
            epoch_bce_loss += bce_loss
            epoch_focal_loss += focal_loss_val
            step_count += 1
            
            if step % 50 == 0:
                print(f"  Step {step}: Loss={total_loss.numpy():.4f}, "
                      f"Acc={train_accuracy.result():.4f}, "
                      f"Prec={train_precision.result():.4f}, "
                      f"Rec={train_recall.result():.4f}")
                
                # Log to wandb
                wandb.log({
                    "real_fake_classifier/step_total_loss": total_loss.numpy(),
                    "real_fake_classifier/step_bce_loss": bce_loss.numpy(),
                    "real_fake_classifier/step_focal_loss": focal_loss_val.numpy(),
                    "real_fake_classifier/step_accuracy": train_accuracy.result().numpy(),
                    "real_fake_classifier/step_precision": train_precision.result().numpy(),
                    "real_fake_classifier/step_recall": train_recall.result().numpy(),
                    "real_fake_classifier/learning_rate": current_lr,
                    "real_fake_classifier/epoch": epoch + 1,
                    "real_fake_classifier/step": step
                })
        
        # Calculate F1 score
        precision = train_precision.result()
        recall = train_recall.result()
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        # Epoch summary
        avg_loss = epoch_loss / step_count
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {train_accuracy.result():.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
        
        # Log epoch metrics
        wandb.log({
            "real_fake_classifier/epoch_avg_loss": avg_loss.numpy(),
            "real_fake_classifier/epoch_avg_bce_loss": (epoch_bce_loss / step_count).numpy(),
            "real_fake_classifier/epoch_avg_focal_loss": (epoch_focal_loss / step_count).numpy(),
            "real_fake_classifier/epoch_accuracy": train_accuracy.result().numpy(),
            "real_fake_classifier/epoch_precision": precision.numpy(),
            "real_fake_classifier/epoch_recall": recall.numpy(),
            "real_fake_classifier/epoch_f1_score": f1_score.numpy(),
            "real_fake_classifier/epoch_completed": epoch + 1
        })
        
        # Save best model
        if epoch == 0 or avg_loss < getattr(train_real_fake_classifier, 'best_loss', float('inf')):
            train_real_fake_classifier.best_loss = avg_loss
            classifier.save_weights("real_fake_classifier.weights.h5")
            print(f"  ✓ Best model saved! Loss: {avg_loss:.6f}")
    
    print("Real/Fake Classifier training completed!")

def evaluate_real_fake_classifier(real_data, encoder, decoder, classifier, num_samples=1000):
    """Evaluate the real/fake classifier performance"""
    print("Evaluating Real/Fake Classifier...")
    
    real_predictions = []
    fake_predictions = []
    
    samples_processed = 0
    for batch in real_data:
        if samples_processed >= num_samples:
            break
            
        real_images = tf.cast(batch['image'], tf.float32) / 255.0
        batch_size = min(tf.shape(real_images)[0], num_samples - samples_processed)
        real_images = real_images[:batch_size]
        
        # Generate fake images
        random_z = tf.random.normal((batch_size, 256))
        fake_images = decoder(random_z, training=False)
        
        # Get predictions
        real_pred = classifier(real_images, training=False)
        fake_pred = classifier(fake_images, training=False)
        
        real_predictions.extend(real_pred.numpy())
        fake_predictions.extend(fake_pred.numpy())
        
        samples_processed += batch_size
    
    real_predictions = np.array(real_predictions)
    fake_predictions = np.array(fake_predictions)
    
    # Calculate metrics
    real_accuracy = np.mean(real_predictions > 0.5)
    fake_accuracy = np.mean(fake_predictions < 0.5)
    overall_accuracy = (real_accuracy + fake_accuracy) / 2
    
    print(f"Real Image Accuracy: {real_accuracy:.4f}")
    print(f"Fake Image Accuracy: {fake_accuracy:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Real Predictions Mean: {np.mean(real_predictions):.4f}")
    print(f"Fake Predictions Mean: {np.mean(fake_predictions):.4f}")
    
    # Log to wandb
    wandb.log({
        "real_fake_classifier/eval_real_accuracy": real_accuracy,
        "real_fake_classifier/eval_fake_accuracy": fake_accuracy,
        "real_fake_classifier/eval_overall_accuracy": overall_accuracy,
        "real_fake_classifier/eval_real_pred_mean": np.mean(real_predictions),
        "real_fake_classifier/eval_fake_pred_mean": np.mean(fake_predictions)
    })
    
    return overall_accuracy

def train_gan(data, inverse_classifier, decoder, real_fake_classifier, epochs=10, batch_size=32, latent_dim=256):
    """
    GAN training: Train inverse classifier and decoder to fool the real/fake classifier
    Generator: inverse_classifier + decoder
    Discriminator: real_fake_classifier
    """
    print("Starting GAN training...")
    
    # Reduce batch size for GAN training to avoid OOM
    gan_batch_size = batch_size // 4  # Use quarter of the original batch size
    print(f"Using reduced batch size for GAN training: {gan_batch_size}")
    
    # Optimizers for generator (inverse + decoder) and discriminator
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=wandb.config.gan_generator_lr,
        beta_1=0.5,
        beta_2=0.999
    )
    
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=wandb.config.gan_discriminator_lr,
        beta_1=0.5,
        beta_2=0.999
    )
    
    # Loss functions
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    # Training step for discriminator with gradient accumulation
    @tf.function
    def train_discriminator_step(real_images, fake_images):
        # Use smaller batch to avoid OOM
        real_batch_size = tf.shape(real_images)[0]
        fake_batch_size = tf.shape(fake_images)[0]
        
        # Split into smaller chunks if needed
        chunk_size = gan_batch_size // 2
        total_loss = 0.0
        num_chunks = 0
        
        # Process real images in chunks
        for i in range(0, real_batch_size, chunk_size):
            end_idx = tf.minimum(i + chunk_size, real_batch_size)
            real_chunk = real_images[i:end_idx]
            
            # Process corresponding fake images
            fake_end_idx = tf.minimum(i + chunk_size, fake_batch_size)
            fake_chunk = fake_images[i:fake_end_idx]
            
            # Combine real and fake chunks
            chunk_batch_size = tf.shape(real_chunk)[0]
            images = tf.concat([real_chunk, fake_chunk], axis=0)
            
            # Labels: 1 for real, 0 for fake
            real_labels = tf.ones((chunk_batch_size, 1), dtype=tf.float32) * 0.9  # Label smoothing
            fake_labels = tf.zeros((chunk_batch_size, 1), dtype=tf.float32) + 0.1  # Label smoothing
            labels = tf.concat([real_labels, fake_labels], axis=0)
            
            with tf.GradientTape() as tape:
                predictions = real_fake_classifier(images, training=True)
                chunk_loss = bce_loss(labels, predictions)
            
            gradients = tape.gradient(chunk_loss, real_fake_classifier.trainable_weights)
            gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
            discriminator_optimizer.apply_gradients(zip(gradients, real_fake_classifier.trainable_weights))
            
            total_loss += chunk_loss
            num_chunks += 1
        
        return total_loss / tf.cast(num_chunks, tf.float32)
    
    # Training step for generator (inverse classifier + decoder)
    @tf.function
    def train_generator_step(attributes):
        # Use smaller batch for generator training
        batch_size_tensor = tf.shape(attributes)[0]
        chunk_size = gan_batch_size // 2
        
        total_loss = 0.0
        total_adv_loss = 0.0
        total_struct_loss = 0.0
        total_fm_loss = 0.0
        num_chunks = 0
        
        for i in range(0, batch_size_tensor, chunk_size):
            end_idx = tf.minimum(i + chunk_size, batch_size_tensor)
            attr_chunk = attributes[i:end_idx]
            
            with tf.GradientTape() as tape:
                # Generate latent codes from attributes
                latent_codes = inverse_classifier(attr_chunk, training=True)
                
                # Generate images from latent codes
                fake_images = decoder(latent_codes, training=True)
                
                # Try to fool discriminator (want discriminator to classify as real)
                fake_predictions = real_fake_classifier(fake_images, training=False)
                real_labels = tf.ones_like(fake_predictions)  # Want to be classified as real
                
                # Generator loss: fooling discriminator + perceptual loss
                adversarial_loss = bce_loss(real_labels, fake_predictions)
                
                # Perceptual loss to maintain image quality (compare with a moving average of real features)
                # For simplicity, we'll use a structural loss instead
                structural_loss = tf.reduce_mean(tf.abs(fake_images - 0.5))  # Encourage natural image distribution
                
                # Feature matching loss
                feature_matching_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(fake_predictions) - 0.5))
                
                # Total generator loss
                chunk_loss = adversarial_loss + 0.1 * structural_loss + 0.01 * feature_matching_loss
            
            # Update both inverse classifier and decoder
            trainable_vars = inverse_classifier.trainable_weights + decoder.trainable_weights
            gradients = tape.gradient(chunk_loss, trainable_vars)
            gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
            generator_optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            total_loss += chunk_loss
            total_adv_loss += adversarial_loss
            total_struct_loss += structural_loss
            total_fm_loss += feature_matching_loss
            num_chunks += 1
        
        num_chunks_float = tf.cast(num_chunks, tf.float32)
        return (total_loss / num_chunks_float, 
                total_adv_loss / num_chunks_float, 
                total_struct_loss / num_chunks_float, 
                total_fm_loss / num_chunks_float)
    
    # Training metrics
    d_loss_metric = tf.keras.metrics.Mean(name='discriminator_loss')
    g_loss_metric = tf.keras.metrics.Mean(name='generator_loss')
    adv_loss_metric = tf.keras.metrics.Mean(name='adversarial_loss')
    
    for epoch in range(epochs):
        print(f"GAN Epoch {epoch + 1}/{epochs}")
        
        # Reset metrics
        d_loss_metric.reset_state()
        g_loss_metric.reset_state()
        adv_loss_metric.reset_state()
        
        step_count = 0
        for step, batch in enumerate(data):
            # Take only part of the batch to reduce memory usage
            real_images = tf.cast(batch['image'], tf.float32) / 255.0
            current_batch_size = tf.shape(real_images)[0]
            
            # Only process if we have enough samples
            if current_batch_size < gan_batch_size:
                continue
                
            # Take only the reduced batch size
            real_images = real_images[:gan_batch_size]
            
            # Convert nested attributes dict to tensor array
            attr_list = []
            for attr_name in sorted(batch['attributes'].keys()):
                attr_values = tf.cast(batch['attributes'][attr_name], tf.float32)
                attr_list.append(attr_values[:gan_batch_size])  # Take only reduced batch
            attributes = tf.stack(attr_list, axis=1)
            
            # Add noise to attributes for diversity
            noise = tf.random.normal(tf.shape(attributes), stddev=0.1)
            attributes_noisy = tf.clip_by_value(attributes + noise, 0.0, 1.0)
            
            # Generate fake images
            try:
                latent_codes = inverse_classifier(attributes_noisy, training=False)
                fake_images = decoder(latent_codes, training=False)
                
                # Train discriminator every step
                d_loss = train_discriminator_step(real_images, fake_images)
                d_loss_metric.update_state(d_loss)
                
                # Train generator every 2 steps (slower generator training)
                if step % 2 == 0:
                    g_loss, adv_loss, struct_loss, fm_loss = train_generator_step(attributes_noisy)
                    g_loss_metric.update_state(g_loss)
                    adv_loss_metric.update_state(adv_loss)
                
                step_count += 1
                
                if step % 100 == 0:
                    print(f"  Step {step}: D_Loss={d_loss:.4f}, G_Loss={g_loss_metric.result():.4f}")
                    
                    # Log to wandb
                    wandb.log({
                        "gan/step_discriminator_loss": d_loss.numpy(),
                        "gan/step_generator_loss": g_loss_metric.result().numpy(),
                        "gan/step_adversarial_loss": adv_loss_metric.result().numpy(),
                        "gan/epoch": epoch + 1,
                        "gan/step": step
                    })
                    
            except tf.errors.ResourceExhaustedError as e:
                print(f"OOM error at step {step}, skipping batch...")
                continue
                
            # Break after processing enough steps to avoid memory buildup
            if step_count >= 500:  # Limit steps per epoch
                break
        
        # Epoch summary
        print(f"GAN Epoch {epoch + 1} Summary:")
        print(f"  Discriminator Loss: {d_loss_metric.result():.4f}")
        print(f"  Generator Loss: {g_loss_metric.result():.4f}")
        print(f"  Adversarial Loss: {adv_loss_metric.result():.4f}")
        
        # Log epoch metrics
        wandb.log({
            "gan/epoch_discriminator_loss": d_loss_metric.result().numpy(),
            "gan/epoch_generator_loss": g_loss_metric.result().numpy(),
            "gan/epoch_adversarial_loss": adv_loss_metric.result().numpy(),
            "gan/epoch_completed": epoch + 1
        })
        
        # Save models every few epochs
        if (epoch + 1) % 5 == 0:
            inverse_classifier.save_weights("inverse_classifier.weights.h5")
            decoder.save_weights("decoder.weights.h5")
            real_fake_classifier.save_weights("real_fake_classifier.weights.h5")
            print(f"  ✓ GAN models saved at epoch {epoch + 1}")
        
        # Generate sample images and log to wandb
        if (epoch + 1) % 2 == 0:
            log_generated_images_to_wandb(inverse_classifier, decoder, f"gan_epoch_{epoch+1}_")
    
    print("GAN training completed!")
    
    # Save final models
    inverse_classifier.save_weights("inverse_classifier.weights.h5")
    decoder.save_weights("decoder.weights.h5")
    real_fake_classifier.save_weights("real_fake_classifier.weights.h5")

if __name__ == "__main__":
    latent_dim = 256
    batch_size = 64 
    epochs = 1      # More epochs

    # Update wandb config with actual values - allow changes for testing
    wandb.config.update({
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "epochs": epochs
    }, allow_val_change=True)

    # Prepare the dataset - keep attributes in the data
    ds_train = ds_train.map(lambda x: {
        'image': tf.image.resize(x['image'], (256, 256)),
        'attributes': x['attributes']
    })
    ds_train = ds_train.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # check if decoder and encoder weights exist, if so load them
    import os
    if os.path.exists("encoder.weights.h5") and os.path.exists("decoder.weights.h5"):
        print("Loading existing model weights...")
        encoder = buildEncoder(latent_dim)
        decoder = buildDecoder(latent_dim)
        encoder.load_weights("encoder.weights.h5")
        decoder.load_weights("decoder.weights.h5")
    else:
        print("No existing model weights found. Training from scratch.")
        # Build models
        encoder = buildEncoder(latent_dim)
        decoder = buildDecoder(latent_dim)

    # Log model summaries to wandb
    print("Logging model architectures to wandb...")
    
    # Capture encoder summary
    encoder_summary = []
    encoder.summary()
    encoder.summary(print_fn=lambda x: encoder_summary.append(x))
    encoder_summary_text = '\n'.join(encoder_summary)
    
    # Capture decoder summary  
    decoder_summary = []
    decoder.summary()
    decoder.summary(print_fn=lambda x: decoder_summary.append(x))
    decoder_summary_text = '\n'.join(decoder_summary)
    
    # Log encoder info
    wandb.log({
        "model_info/encoder_params": encoder.count_params(),
        "model_info/encoder_trainable_params": sum([tf.size(w).numpy() for w in encoder.trainable_weights]),
        "model_info/encoder_layers": len(encoder.layers)
    })
    
    # Log decoder info
    wandb.log({
        "model_info/decoder_params": decoder.count_params(),
        "model_info/decoder_trainable_params": sum([tf.size(w).numpy() for w in decoder.trainable_weights]),
        "model_info/decoder_layers": len(decoder.layers)
    })
    
    # Log architecture summaries as text
    wandb.log({
        "model_summaries/encoder": wandb.Html(f"<pre>{encoder_summary_text}</pre>"),
        "model_summaries/decoder": wandb.Html(f"<pre>{decoder_summary_text}</pre>")
    })
    
    display_reconstructions(encoder, decoder, ds_train, num_images=1)

    classifier = buildClassifier(latent_dim)
    if os.path.exists("classifier.weights.h5"):
        print("Loading existing classifier weights...")
        classifier.load_weights("classifier.weights.h5")

    # Log classifier info
    classifier_summary = []
    classifier.summary()
    classifier.summary(print_fn=lambda x: classifier_summary.append(x))
    classifier_summary_text = '\n'.join(classifier_summary)
    
    wandb.log({
        "model_info/classifier_params": classifier.count_params(),
        "model_info/classifier_trainable_params": sum([tf.size(w).numpy() for w in classifier.trainable_weights]),
        "model_info/classifier_layers": len(classifier.layers),
        "model_summaries/classifier": wandb.Html(f"<pre>{classifier_summary_text}</pre>")
    })

    # Build and train inverse classifier
    inverse_classifier = buildInverseClassifier(latent_dim)
    if os.path.exists("inverse_classifier.weights.h5"):
        print("Loading existing inverse classifier weights...")
        inverse_classifier.load_weights("inverse_classifier.weights.h5")
    
    # Log inverse classifier info
    inverse_summary = []
    inverse_classifier.summary()
    inverse_classifier.summary(print_fn=lambda x: inverse_summary.append(x))
    inverse_summary_text = '\n'.join(inverse_summary)
    
    wandb.log({
        "model_info/inverse_classifier_params": inverse_classifier.count_params(),
        "model_info/inverse_classifier_trainable_params": sum([tf.size(w).numpy() for w in inverse_classifier.trainable_weights]),
        "model_info/inverse_classifier_layers": len(inverse_classifier.layers),
        "model_summaries/inverse_classifier": wandb.Html(f"<pre>{inverse_summary_text}</pre>")
    })
    
    # Log total parameter count for all models
    total_params = (encoder.count_params() + decoder.count_params() + 
                   classifier.count_params() + inverse_classifier.count_params())
    
    wandb.log({
        "model_info/total_params_all_models": total_params,
        "model_info/vae_params": encoder.count_params() + decoder.count_params()
    })
    
    print(f"Total parameters across all models: {total_params:,}")
    
    display_reconstructions(encoder, decoder, ds_train, num_images=1)

    classifier = buildClassifier(latent_dim)
    if os.path.exists("classifier.weights.h5"):
        print("Loading existing classifier weights...")
        classifier.load_weights("classifier.weights.h5")

    # Build and train inverse classifier
    inverse_classifier = buildInverseClassifier(latent_dim)
    if os.path.exists("inverse_classifier.weights.h5"):
        print("Loading existing inverse classifier weights...")
        inverse_classifier.load_weights("inverse_classifier.weights.h5")
    
    desired_features = ['Young', 'Smiling', 'Attractive', 'Male']
    intensities = [1.0, 1.0, 1.0, 0.0]
    feature_vector = create_feature_vector_from_descriptions(desired_features, intensities=intensities)
    generated_image = generate_image_from_features(feature_vector, inverse_classifier, decoder)

    # Log initial generation to wandb
    text = ''
    for feat, inten in zip(desired_features, intensities):
        text += f"{feat}: {inten:.2f}, "
    text = text.strip().rstrip(',')
    wandb.log({
        "initial_generation": wandb.Image(np.clip(generated_image, 0, 1), 
                                         caption=f"Initial: {text}")
    })

    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.clip(generated_image, 0, 1))
    plt.title(f"Strong: {', '.join(desired_features)}")
    plt.axis('off')

    plt.show()

    desired_features = ['Young', 'Smiling', 'Blond_Hair', 'Attractive', 'Male']
    intensities = [0.1, 0.21, 0.95, 0.8, 0.9]
    feature_vector = create_feature_vector_from_descriptions(desired_features, intensities=intensities)
    generated_image = generate_image_from_features(feature_vector, inverse_classifier, decoder)

    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.clip(generated_image, 0, 1))
    plt.title(f"Strong: {', '.join(desired_features)}")
    plt.axis('off')

    plt.show()

    # Training with wandb logging
    print("Starting training with wandb logging...")
    
    # train(ds_train, encoder, decoder, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
    
    # Log sample reconstructions after VAE training
    log_generated_images_to_wandb(inverse_classifier, decoder, "post_vae_")
    
    train_inverse_classifier(ds_train, encoder, inverse_classifier, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
    
    # Log sample generations after inverse classifier training
    log_generated_images_to_wandb(inverse_classifier, decoder, "post_inverse_")
    
    train_classifier(ds_train, encoder, classifier, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
    
    # Build and train real/fake classifier
    print("\n=== Training Real/Fake Classifier ===")
    real_fake_classifier = buildRealFakeClassifier()
    if os.path.exists("real_fake_classifier.weights.h5"):
        print("Loading existing real/fake classifier weights...")
        real_fake_classifier.load_weights("real_fake_classifier.weights.h5")
    else:
        train_real_fake_classifier(ds_train, encoder, decoder, real_fake_classifier, epochs=1, batch_size=batch_size)
    
    real_fake_classifier.summary()
    real_fake_classifier_summary = []
    real_fake_classifier.summary(print_fn=lambda x: real_fake_classifier_summary.append(x))
    real_fake_classifier_summary_text = '\n'.join(real_fake_classifier_summary)
    wandb.log({
        "model_info/real_fake_classifier_params": real_fake_classifier.count_params(),
        "model_info/real_fake_classifier_trainable_params": sum([tf.size(w).numpy() for w in real_fake_classifier.trainable_weights]),
        "model_info/real_fake_classifier_layers": len(real_fake_classifier.layers),
        "model_summaries/real_fake_classifier": wandb.Html(f"<pre>{real_fake_classifier_summary_text}</pre>")
    })

    # Evaluate real/fake classifier
    accuracy = evaluate_real_fake_classifier(ds_train, encoder, decoder, real_fake_classifier, num_samples=1000)
    print(f"Real/Fake Classifier Accuracy: {accuracy:.4f}")
    
    # GAN Training: Use real/fake classifier to improve generator (inverse classifier + decoder)
    print("\n=== Starting GAN Training ===")
    gan_epochs = wandb.config.gan_epochs
    # Use smaller batch size for GAN training to avoid OOM
    gan_batch_size = 16  # Reduced from original batch_size
    train_gan(ds_train, inverse_classifier, decoder, real_fake_classifier, epochs=gan_epochs, batch_size=gan_batch_size, latent_dim=latent_dim)
    
    # Log post-GAN generations
    log_generated_images_to_wandb(inverse_classifier, decoder, "post_gan_")

    print("\n=== Generating Images from Features ===")
    
    desired_features = ['Young', 'Smiling', 'Blond_Hair', 'Attractive']
    intensities = [0.8, 0.9, 0.7, 0.8]
    feature_vector = create_feature_vector_from_descriptions(desired_features, intensities=intensities)
    
    generated_image = generate_image_from_features(feature_vector, inverse_classifier, decoder)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.clip(generated_image, 0, 1))
    plt.title(f"Strong: {', '.join(desired_features)}")
    plt.axis('off')
    
    # Subtle version
    intensities_subtle = [0.3, 0.4, 0.2, 0.3]
    feature_vector_subtle = create_feature_vector_from_descriptions(desired_features, intensities=intensities_subtle)
    generated_image_subtle = generate_image_from_features(feature_vector_subtle, inverse_classifier, decoder)
    
    plt.subplot(2, 3, 2)
    plt.imshow(np.clip(generated_image_subtle, 0, 1))
    plt.title(f"Subtle: {', '.join(desired_features)}")
    plt.axis('off')
    
    # Male with varying intensities
    desired_features2 = ['Male', 'Eyeglasses', 'Mustache', 'Young']
    intensities2 = [0.9, 0.8, 0.6, 0.7]
    feature_vector2 = create_feature_vector_from_descriptions(desired_features2, intensities=intensities2)
    generated_image2 = generate_image_from_features(feature_vector2, inverse_classifier, decoder)
    
    plt.subplot(2, 3, 3)
    plt.imshow(np.clip(generated_image2, 0, 1))
    plt.title(f"Male Strong")
    plt.axis('off')
    
    # Progressive smile intensity
    for i, smile_intensity in enumerate([0.2, 0.5, 0.8]):
        features = ['Young', 'Smiling', 'Attractive']
        intensities = [0.8, smile_intensity, 0.7]
        fv = create_feature_vector_from_descriptions(features, intensities=intensities)
        img = generate_image_from_features(fv, inverse_classifier, decoder)
        
        plt.subplot(2, 3, 4 + i)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"Smile: {smile_intensity}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display some reconstructions
    display_reconstructions(encoder, decoder, ds_train, num_images=1)
    
    # Log final results
    log_generated_images_to_wandb(inverse_classifier, decoder, "final_")
    
    # Finish wandb run
    wandb.finish()