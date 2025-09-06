import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Input
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import wandb
import io
import os
import datetime

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
        "gan_epochs": 30,
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
    
    # Simpler initial projection
    x = Dense(8 * 8 * 512, activation="relu")(decoder_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Reshape((8, 8, 512))(x)
    
    # Upsampling blocks
    x = Conv2DTranspose(256, 4, strides=2, padding="same")(x)  # 16x16
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(128, 4, strides=2, padding="same")(x)  # 32x32
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(64, 4, strides=2, padding="same")(x)  # 64x64
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(32, 4, strides=2, padding="same")(x)  # 128x128
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(16, 4, strides=2, padding="same")(x)  # 256x256
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Final output layer
    decoder_output = Conv2D(3, 3, activation="sigmoid", padding="same", name="decoder_output")(x)
    
    decoder = Model(decoder_input, decoder_output, name="simplified_decoder")
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
        
        # Log reconstruction images every few epochs
        if (epoch + 1) % 5 == 0:
            print(f"  Logging reconstruction images for epoch {epoch + 1}")
            log_reconstruction_images_to_wandb(encoder, decoder, data, f"vae_epoch_{epoch+1}_")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"   New best model saved! Loss: {best_loss:.6f}")
            encoder.save_weights("encoder.weights.h5")
            decoder.save_weights("decoder.weights.h5")
            wandb.log({
                "vae/new_best_loss": best_loss.numpy(),
                "vae/best_epoch": epoch + 1
            })
    
    print("VAE training completed.")
    
    # Log final reconstruction images
    print("Logging final VAE reconstruction images...")
    log_reconstruction_images_to_wandb(encoder, decoder, data, "vae_final_")

def buildInverseClassifier(latent_dim):
    """Simplified network"""
    inverse_input = Input(shape=(40,), name="inverse_input")  # 40 CelebA attributes
    
    x = Dense(512, activation="relu")(inverse_input)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = Dense(1024, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = Dense(1024, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = Dense(512, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    inverse_output = Dense(latent_dim)(x)
    
    inverse_classifier = Model(inverse_input, inverse_output, name="simplified_inverse_classifier")
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
            print(f"   New best model saved! Loss: {best_loss:.6f}")
            
            # Log best model update to wandb
            wandb.log({
                "inverse_classifier/new_best_loss": best_loss,
                "inverse_classifier/best_epoch": epoch + 1
            })
        else:
            print(f"  Current best loss: {best_loss:.6f}")
    
    print(f"Ultra enhanced inverse classifier training completed. Best loss: {best_loss:.6f}")

def generate_image_from_features(feature_list, inverse_classifier, decoder, denoiser=None, attribute_names=None):
    features = tf.constant([feature_list], dtype=tf.float32)
    latent_code = inverse_classifier(features, training=False)
    
    # Add slight noise for variation
    noise = tf.random.normal(tf.shape(latent_code), stddev=0.1)
    latent_code = latent_code + noise
    
    # Apply denoiser if available to enhance latent representation
    if denoiser is not None:
        latent_code = denoiser(latent_code, training=False)
    
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

def log_generated_images_to_wandb(inverse_classifier, decoder, denoiser=None, epoch_prefix=""):
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
            generated_image = generate_image_from_features(feature_vector, inverse_classifier, decoder, denoiser)
            
            # Convert to PIL Image for wandb
            generated_image_clipped = np.clip(generated_image, 0, 1)
            images_to_log[f"{epoch_prefix}generated_{name}"] = wandb.Image(generated_image_clipped, caption=f"{name}: {', '.join(features)}")
        
        wandb.log(images_to_log)
        
    except Exception as e:
        print(f"Warning: Could not log images to wandb: {e}")

def log_reconstruction_images_to_wandb(encoder, decoder, data, epoch_prefix="", num_images=4):
    """Log encoder-decoder reconstruction images to wandb"""
    try:
        # Get a batch of real images
        sample_batch = next(iter(data.take(1)))
        real_images = tf.cast(sample_batch['image'], tf.float32) / 255.0
        real_images = real_images[:num_images]
        
        # Get reconstructions
        z_mean, z_log_var, z = encoder(real_images, training=False)
        reconstructions = decoder(z, training=False)
        
        # Prepare images for logging
        images_to_log = {}
        
        for i in range(num_images):
            # Original image
            original = np.clip(real_images[i].numpy(), 0, 1)
            # Reconstructed image
            reconstructed = np.clip(reconstructions[i].numpy(), 0, 1)
            
            # Log both as separate images
            images_to_log[f"{epoch_prefix}original_{i}"] = wandb.Image(original, caption=f"Original {i}")
            images_to_log[f"{epoch_prefix}reconstruction_{i}"] = wandb.Image(reconstructed, caption=f"Reconstruction {i}")
            
            # Create side-by-side comparison
            comparison = np.concatenate([original, reconstructed], axis=1)
            images_to_log[f"{epoch_prefix}comparison_{i}"] = wandb.Image(comparison, caption=f"Original vs Reconstruction {i}")
        
        wandb.log(images_to_log)
        
    except Exception as e:
        print(f"Warning: Could not log reconstruction images to wandb: {e}")


def buildRealFakeClassifier():
    """Simplified real/fake classifier"""
    input_layer = Input(shape=(256, 256, 3), name="real_fake_input")
    
    x = Conv2D(32, 4, strides=2, padding="same")(input_layer)  # 128x128
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(64, 4, strides=2, padding="same")(x)  # 64x64
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = Conv2D(128, 4, strides=2, padding="same")(x)  # 32x32
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = Conv2D(256, 4, strides=2, padding="same")(x)  # 16x16
    x = LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)
    
    model = Model(input_layer, x, name="simplified_real_fake_classifier")
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
            print(f"  Best model saved! Loss: {avg_loss:.6f}")
    
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

def buildDenoiser(latent_dim):
    """Advanced denoiser network with fewer parameters using efficient architectures"""
    denoiser_input = Input(shape=(latent_dim,), name="denoiser_input")
    
    # Residual blocks for better gradient flow with fewer parameters
    def residual_block(x, units, dropout_rate=0.1):
        # Main path
        main = Dense(units, activation=None)(x)
        main = tf.keras.layers.LayerNormalization()(main)
        main = tf.keras.layers.Activation('swish')(main)  # Swish is more efficient than ReLU
        main = tf.keras.layers.Dropout(dropout_rate)(main)
        
        # Skip connection with projection if needed
        if x.shape[-1] != units:
            skip = Dense(units, activation=None)(x)
        else:
            skip = x
            
        return tf.keras.layers.Add()([main, skip])
    
    # Attention mechanism for better feature selection
    def attention_layer(x, units):
        # Simplified channel attention mechanism
        # Squeeze: reduce to single value per channel using dense layer
        squeezed = Dense(units // 8, activation='relu')(x)
        
        # Excitation: generate attention weights
        attention_weights = Dense(units, activation='sigmoid')(squeezed)
        
        # Apply attention weights element-wise
        attended = tf.keras.layers.Multiply()([x, attention_weights])
        return attended
    
    # Efficient progressive refinement with skip connections
    # First expansion with residual connection
    x1 = residual_block(denoiser_input, 384, dropout_rate=0.15)  # Reduced from 512
    
    # Second level with attention
    x2 = residual_block(x1, 512, dropout_rate=0.1)  # Reduced from 1024
    x2_attention = attention_layer(x2, 512)
    x2 = tf.keras.layers.Add()([x2, x2_attention])  # Residual attention
    
    # Third level - bottleneck
    x3 = residual_block(x2, 384, dropout_rate=0.1)  # Reduced from 1024
    
    # Compression back down with skip connections
    x4 = residual_block(x3, 256, dropout_rate=0.1)  # Reduced from 512
    
    # Skip connection from input for better gradient flow
    input_proj = Dense(256, activation=None)(denoiser_input)
    x4 = tf.keras.layers.Add()([x4, input_proj])
    
    # Final output layer with residual connection to input
    denoiser_output = Dense(latent_dim, activation=None)(x4)
    
    # Global skip connection from input to output
    final_output = tf.keras.layers.Add()([denoiser_input, denoiser_output])
    
    denoiser = Model(denoiser_input, final_output, name="efficient_advanced_denoiser")
    
    # Print parameter count
    print(f"Denoiser Parameters: {denoiser.count_params():,}")
    
    return denoiser

def train_denoiser(data, encoder, decoder, denoiser, epochs=10, batch_size=32, latent_dim=256):
    """Train denoiser to remove noise from latent representations"""
    print("Starting Denoiser training...")
    
    # Optimizer with learning rate scheduling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Loss functions
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
    
    # Metrics
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    mse_metric = tf.keras.metrics.Mean(name='mse_loss')
    mae_metric = tf.keras.metrics.Mean(name='mae_loss')
    
    best_loss = float('inf')
    
    @tf.function
    def train_step(images):
        with tf.GradientTape() as tape:
            # Get clean latent representations
            z_mean, z_log_var, z_clean = encoder(images, training=False)
            
            # Add various types of noise to latent space
            batch_size = tf.shape(z_clean)[0]
            
            # 1. Gaussian noise with random variance
            noise_std = tf.random.uniform([batch_size, 1], 0.05, 0.3)
            gaussian_noise = tf.random.normal(tf.shape(z_clean)) * noise_std
            
            # 2. Salt and pepper noise
            salt_pepper_mask = tf.random.uniform(tf.shape(z_clean)) < 0.1
            salt_pepper_noise = tf.where(
                salt_pepper_mask,
                tf.random.uniform(tf.shape(z_clean), -2.0, 2.0),
                tf.zeros_like(z_clean)
            )
            
            # 3. Dropout-like noise (set some values to 0)
            dropout_mask = tf.random.uniform(tf.shape(z_clean)) > 0.85
            dropout_noise = tf.where(dropout_mask, -z_clean, tf.zeros_like(z_clean))
            
            # Combine different noise types
            total_noise = gaussian_noise + salt_pepper_noise + dropout_noise
            z_noisy = z_clean + total_noise
            
            # Denoiser prediction
            z_denoised = denoiser(z_noisy, training=True)
            
            # Multi-objective loss
            mse_loss = mse_loss_fn(z_clean, z_denoised)
            mae_loss = mae_loss_fn(z_clean, z_denoised)
            
            # Consistency loss: denoised latents should produce similar images
            images_from_clean = decoder(z_clean, training=False)
            images_from_denoised = decoder(z_denoised, training=False)
            consistency_loss = tf.reduce_mean(tf.abs(images_from_clean - images_from_denoised))
            
            # L2 regularization
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in denoiser.trainable_weights])
            
            # Combined loss
            total_loss = (0.4 * mse_loss + 
                         0.3 * mae_loss + 
                         0.2 * consistency_loss + 
                         0.0001 * l2_loss)
            
        gradients = tape.gradient(total_loss, denoiser.trainable_weights)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        optimizer.apply_gradients(zip(gradients, denoiser.trainable_weights))
        
        return total_loss, mse_loss, mae_loss, consistency_loss
    
    for epoch in range(epochs):
        print(f"Denoiser Epoch {epoch + 1}/{epochs}")
        
        # Reset metrics
        train_loss_metric.reset_state()
        mse_metric.reset_state()
        mae_metric.reset_state()
        
        for step, batch in enumerate(data):
            images = tf.cast(batch['image'], tf.float32) / 255.0
            
            total_loss, mse_loss, mae_loss, consistency_loss = train_step(images)
            
            # Update metrics
            train_loss_metric.update_state(total_loss)
            mse_metric.update_state(mse_loss)
            mae_metric.update_state(mae_loss)
            
            if step % 100 == 0:
                print(f"  Step {step}: Total={total_loss.numpy():.6f}, "
                      f"MSE={mse_loss.numpy():.6f}, MAE={mae_loss.numpy():.6f}, "
                      f"Consistency={consistency_loss.numpy():.6f}")
                
                wandb.log({
                    "denoiser/step_total_loss": total_loss.numpy(),
                    "denoiser/step_mse_loss": mse_loss.numpy(),
                    "denoiser/step_mae_loss": mae_loss.numpy(),
                    "denoiser/step_consistency_loss": consistency_loss.numpy(),
                    "denoiser/learning_rate": optimizer.learning_rate.numpy(),
                    "denoiser/epoch": epoch + 1,
                    "denoiser/step": step
                })
        
        # Epoch summary
        epoch_avg_loss = train_loss_metric.result().numpy()
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Total Loss: {epoch_avg_loss:.6f}")
        print(f"  Average MSE Loss: {mse_metric.result():.6f}")
        print(f"  Average MAE Loss: {mae_metric.result():.6f}")
        
        wandb.log({
            "denoiser/epoch_avg_total_loss": epoch_avg_loss,
            "denoiser/epoch_avg_mse_loss": mse_metric.result().numpy(),
            "denoiser/epoch_avg_mae_loss": mae_metric.result().numpy(),
            "denoiser/epoch_completed": epoch + 1
        })
        
        # Save best model
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            denoiser.save_weights("denoiser.weights.h5")
            print(f" New best denoiser model saved! Loss: {best_loss:.6f}")
            
            wandb.log({
                "denoiser/new_best_loss": best_loss,
                "denoiser/best_epoch": epoch + 1
            })
    
    print(f"Denoiser training completed. Best loss: {best_loss:.6f}")
    return denoiser

def evaluate_reconstruction_quality(encoder, decoder, data, num_samples=100):
    """Evaluate reconstruction quality with multiple metrics"""
    ssim_values = []
    psnr_values = []
    recon_losses = []
    
    samples_processed = 0
    for batch in data:
        if samples_processed >= num_samples:
            break
            
        real_images = tf.cast(batch['image'], tf.float32) / 255.0
        batch_size = min(tf.shape(real_images)[0], num_samples - samples_processed)
        real_images = real_images[:batch_size]
        
        # Get reconstructions
        z_mean, z_log_var, z = encoder(real_images, training=False)
        reconstructions = decoder(z, training=False)
        
        # Compute metrics
        ssim_batch = tf.image.ssim(real_images, reconstructions, max_val=1.0)
        ssim_values.extend(ssim_batch.numpy())
        
        mse_batch = tf.reduce_mean(tf.square(real_images - reconstructions), axis=[1,2,3])
        psnr_batch = 20 * tf.math.log(1.0 / tf.sqrt(mse_batch + 1e-8)) / tf.math.log(10.0)
        psnr_values.extend(psnr_batch.numpy())
        
        recon_loss_batch = tf.reduce_mean(tf.abs(real_images - reconstructions), axis=[1,2,3])
        recon_losses.extend(recon_loss_batch.numpy())
        
        samples_processed += batch_size
    
    return {
        'ssim': np.mean(ssim_values),
        'psnr': np.mean(psnr_values),
        'recon_loss': np.mean(recon_losses),
        'ssim_std': np.std(ssim_values),
        'psnr_std': np.std(psnr_values)
    }

def train_gan(data, inverse_classifier, decoder, real_fake_classifier, encoder, denoiser=None, epochs=10, batch_size=32, latent_dim=256):
    """
    Enhanced GAN training with stronger quality preservation and denoiser integration
    """
    print("Starting Enhanced GAN training...")
    if denoiser is not None:
        print("Using denoiser for latent space enhancement")
    
    gan_batch_size = batch_size // 2  # Increased from //4 to //2 for better training
    print(f"Using batch size for GAN training: {gan_batch_size}")
    
    # Optimizers with more conservative learning rates
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001, beta_1=0.5, beta_2=0.999  # Reduced from 0.0002
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001, beta_1=0.5, beta_2=0.999  # Increased from 0.0001 to match
    )
    
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    # VGG for perceptual loss
    vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    vgg.trainable = False
    
    # Quality metrics
    ssim_metric = tf.keras.metrics.Mean(name='ssim')
    psnr_metric = tf.keras.metrics.Mean(name='psnr')
    reconstruction_metric = tf.keras.metrics.Mean(name='reconstruction_loss')

    # Best model tracking for periodic saving
    best_combined_score = float('inf')  # Combined SSIM + Generator performance
    best_epoch = 0
    patience = 8  # Early stopping patience
    no_improvement_count = 0
    save_frequency = 3  # Save best model every N epochs

    @tf.function
    def compute_perceptual_loss(real_imgs, fake_imgs):
        real_features = vgg(real_imgs * 255.0)
        fake_features = vgg(fake_imgs * 255.0)
        return tf.reduce_mean(tf.abs(real_features - fake_features))

    @tf.function
    def compute_quality_metrics(real_imgs, recon_imgs):
        # SSIM
        ssim_val = tf.reduce_mean(tf.image.ssim(real_imgs, recon_imgs, max_val=1.0))
        
        # PSNR
        mse = tf.reduce_mean(tf.square(real_imgs - recon_imgs))
        psnr_val = tf.constant(20.0) * tf.math.log(tf.constant(1.0) / tf.sqrt(mse + 1e-8)) / tf.math.log(tf.constant(10.0))
        
        return ssim_val, psnr_val

    @tf.function
    def train_discriminator_step(real_images, fake_images):
        with tf.GradientTape() as tape:
            real_output = real_fake_classifier(real_images, training=True)
            fake_output = real_fake_classifier(fake_images, training=True)
            
            # More aggressive label smoothing for better training
            real_loss = bce_loss(tf.ones_like(real_output) * 0.85, real_output)  # More smoothing
            fake_loss = bce_loss(tf.zeros_like(fake_output) + 0.15, fake_output)  # More smoothing
            total_loss = real_loss + fake_loss
            
        gradients = tape.gradient(total_loss, real_fake_classifier.trainable_weights)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        discriminator_optimizer.apply_gradients(zip(gradients, real_fake_classifier.trainable_weights))
        return total_loss

    @tf.function
    def train_generator_step(attributes, real_images):
        with tf.GradientTape() as tape:
            # Generate latent codes from attributes
            latent_codes_raw = inverse_classifier(attributes, training=True)
            
            # Apply denoiser to enhance latent codes if available
            if denoiser is not None:
                latent_codes = denoiser(latent_codes_raw, training=False)  # Don't train denoiser in GAN
                
                # Denoiser quality loss - encourage raw codes to be clean
                denoiser_loss = tf.reduce_mean(tf.abs(latent_codes_raw - latent_codes))
            else:
                latent_codes = latent_codes_raw
                denoiser_loss = 0.0
            
            # Generate fake images from (potentially denoised) latent codes
            fake_images = decoder(latent_codes, training=True)
            
            # Adversarial loss
            fake_predictions = real_fake_classifier(fake_images, training=False)
            adversarial_loss = bce_loss(tf.ones_like(fake_predictions), fake_predictions)
            
            # Enhanced reconstruction losses
            z_mean, z_log_var, z_real = encoder(real_images, training=False)
            
            # 1. Latent space consistency (compare with denoised codes)
            latent_consistency_loss = tf.reduce_mean(tf.abs(latent_codes - z_mean))
            
            # 2. Raw latent consistency (encourage inverse classifier to generate clean codes)
            raw_latent_consistency_loss = tf.reduce_mean(tf.abs(latent_codes_raw - z_mean))
            
            # 3. Image reconstruction loss (L1 + L2)
            l1_loss = tf.reduce_mean(tf.abs(fake_images - real_images))
            l2_loss = tf.reduce_mean(tf.square(fake_images - real_images))
            pixel_loss = 0.7 * l1_loss + 0.3 * l2_loss
            
            # 4. Perceptual loss
            perceptual_loss = compute_perceptual_loss(real_images, fake_images)
            
            # 5. Round-trip consistency (encoder -> decoder -> encoder)
            z_fake_mean, _, _ = encoder(fake_images, training=False)
            cycle_loss = tf.reduce_mean(tf.abs(z_fake_mean - latent_codes))
            
            # Combined loss with more balanced weights and denoiser integration
            if denoiser is not None:
                total_loss = (0.8 * adversarial_loss +           # Reduced adversarial weight
                             0.8 * latent_consistency_loss +      # Reduced weight since we have raw consistency
                             0.6 * raw_latent_consistency_loss +  # Encourage clean raw generation
                             4.0 * pixel_loss +                  # Increased image quality weight
                             0.8 * perceptual_loss +             # Increased perceptual quality
                             0.5 * cycle_loss +                  # Reduced cycle consistency
                             0.3 * denoiser_loss)                # Denoiser quality
            else:
                total_loss = (1.0 * adversarial_loss + 
                             1.0 * latent_consistency_loss +     # Reduced weight
                             5.0 * pixel_loss +                 # Much higher image quality focus
                             1.0 * perceptual_loss +            # Increased perceptual quality
                             0.5 * cycle_loss)                  # Reduced cycle consistency
                raw_latent_consistency_loss = 0.0
            
        # Train both inverse classifier AND decoder
        trainable_vars = inverse_classifier.trainable_weights + decoder.trainable_weights
        gradients = tape.gradient(total_loss, trainable_vars)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        generator_optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return total_loss, adversarial_loss, latent_consistency_loss, pixel_loss, perceptual_loss, cycle_loss, raw_latent_consistency_loss, denoiser_loss

    # Track initial quality
    print("Computing initial reconstruction quality...")
    initial_quality = evaluate_reconstruction_quality(encoder, decoder, data, num_samples=100)
    wandb.log({
        "gan/initial_ssim": initial_quality['ssim'],
        "gan/initial_psnr": initial_quality['psnr'],
        "gan/initial_reconstruction_loss": initial_quality['recon_loss']
    })

    for epoch in range(epochs):
        print(f"GAN Epoch {epoch + 1}/{epochs}")
        
        # Reset metrics
        ssim_metric.reset_state()
        psnr_metric.reset_state()
        reconstruction_metric.reset_state()
        
        d_loss_metric = tf.keras.metrics.Mean(name='d_loss')
        g_loss_metric = tf.keras.metrics.Mean(name='g_loss')
        
        epoch_adv_loss = tf.keras.metrics.Mean(name='adv_loss')
        epoch_latent_loss = tf.keras.metrics.Mean(name='latent_loss')
        epoch_pixel_loss = tf.keras.metrics.Mean(name='pixel_loss')
        epoch_perceptual_loss = tf.keras.metrics.Mean(name='perceptual_loss')
        epoch_cycle_loss = tf.keras.metrics.Mean(name='cycle_loss')
        epoch_raw_latent_loss = tf.keras.metrics.Mean(name='raw_latent_loss')
        epoch_denoiser_quality_loss = tf.keras.metrics.Mean(name='denoiser_quality_loss')

        for step, batch in enumerate(data):
            real_images = tf.cast(batch['image'], tf.float32) / 255.0
            
            attr_list = []
            for attr_name in sorted(batch['attributes'].keys()):
                attr_list.append(tf.cast(batch['attributes'][attr_name], tf.float32))
            attributes = tf.stack(attr_list, axis=1)
            
            # Add noise to attributes for diversity
            attributes_noisy = tf.clip_by_value(
                attributes + tf.random.normal(tf.shape(attributes), stddev=0.05), 
                0.0, 1.0
            )

            # Train generator
            g_loss, adv_loss, latent_loss, pixel_loss, perc_loss, cycle_loss, raw_latent_loss, denoiser_quality_loss = train_generator_step(
                attributes_noisy, real_images
            )
            
            # Update loss metrics
            g_loss_metric.update_state(g_loss)
            epoch_adv_loss.update_state(adv_loss)
            epoch_latent_loss.update_state(latent_loss)
            epoch_pixel_loss.update_state(pixel_loss)
            epoch_perceptual_loss.update_state(perc_loss)
            epoch_cycle_loss.update_state(cycle_loss)
            epoch_raw_latent_loss.update_state(raw_latent_loss)
            epoch_denoiser_quality_loss.update_state(denoiser_quality_loss)
            
            # Compute quality metrics with denoised latent codes
            if denoiser is not None:
                latent_codes_raw = inverse_classifier(attributes_noisy, training=False)
                latent_codes = denoiser(latent_codes_raw, training=False)
                fake_images = decoder(latent_codes, training=False)
            else:
                latent_codes = inverse_classifier(attributes_noisy, training=False)
                fake_images = decoder(latent_codes, training=False)
            
            ssim_val, psnr_val = compute_quality_metrics(real_images, fake_images)
            recon_loss_val = tf.reduce_mean(tf.abs(real_images - fake_images))
            
            ssim_metric.update_state(ssim_val)
            psnr_metric.update_state(psnr_val)
            reconstruction_metric.update_state(recon_loss_val)
            
            # Train discriminator less frequently to prevent it from overpowering generator
            if step % 3 == 0:  # Train discriminator every 3 steps instead of every 2
                d_loss = train_discriminator_step(real_images, fake_images)
                d_loss_metric.update_state(d_loss)
            
            if step % 50 == 0:
                if denoiser is not None:
                    print(f"  Step {step}: D_Loss={d_loss_metric.result():.4f}, "
                          f"G_Loss={g_loss_metric.result():.4f}, "
                          f"SSIM={ssim_metric.result():.4f}, "
                          f"PSNR={psnr_metric.result():.2f}, "
                          f"Denoiser_Quality={epoch_denoiser_quality_loss.result():.6f}")
                else:
                    print(f"  Step {step}: D_Loss={d_loss_metric.result():.4f}, "
                          f"G_Loss={g_loss_metric.result():.4f}, "
                          f"SSIM={ssim_metric.result():.4f}, "
                          f"PSNR={psnr_metric.result():.2f}")
                
                wandb_log_dict = {
                    "gan/step_discriminator_loss": d_loss_metric.result(),
                    "gan/step_generator_loss": g_loss_metric.result(),
                    "gan/step_adversarial_loss": epoch_adv_loss.result(),
                    "gan/step_latent_consistency_loss": epoch_latent_loss.result(),
                    "gan/step_pixel_loss": epoch_pixel_loss.result(),
                    "gan/step_perceptual_loss": epoch_perceptual_loss.result(),
                    "gan/step_cycle_loss": epoch_cycle_loss.result(),
                    "gan/step_ssim": ssim_metric.result(),
                    "gan/step_psnr": psnr_metric.result(),
                    "gan/step_reconstruction_loss": reconstruction_metric.result(),
                    "gan/epoch": epoch + 1,
                    "gan/step": step
                }
                
                if denoiser is not None:
                    wandb_log_dict.update({
                        "gan/step_raw_latent_consistency_loss": epoch_raw_latent_loss.result(),
                        "gan/step_denoiser_quality_loss": epoch_denoiser_quality_loss.result()
                    })
                
                wandb.log(wandb_log_dict)

        # Epoch summary
        print(f"GAN Epoch {epoch + 1} Summary:")
        print(f"  D_Loss: {d_loss_metric.result():.4f}")
        print(f"  G_Loss: {g_loss_metric.result():.4f}")
        print(f"  SSIM: {ssim_metric.result():.4f}")
        print(f"  PSNR: {psnr_metric.result():.2f}")
        print(f"  Reconstruction Loss: {reconstruction_metric.result():.6f}")
        
        if denoiser is not None:
            print(f"  Raw Latent Consistency: {epoch_raw_latent_loss.result():.6f}")
            print(f"  Denoiser Quality Loss: {epoch_denoiser_quality_loss.result():.6f}")
        
        # Check for quality degradation
        if epoch > 0:
            current_ssim = ssim_metric.result().numpy()
            if hasattr(train_gan, 'best_ssim'):
                if current_ssim < train_gan.best_ssim - 0.1:  # Significant drop
                    print(f"  WARNING: SSIM dropped from {train_gan.best_ssim:.4f} to {current_ssim:.4f}")
                train_gan.best_ssim = max(train_gan.best_ssim, current_ssim)
            else:
                train_gan.best_ssim = current_ssim
        else:
            train_gan.best_ssim = ssim_metric.result().numpy()

        wandb_epoch_log = {
            "gan/epoch_discriminator_loss": d_loss_metric.result(),
            "gan/epoch_generator_loss": g_loss_metric.result(),
            "gan/epoch_adversarial_loss": epoch_adv_loss.result(),
            "gan/epoch_latent_consistency_loss": epoch_latent_loss.result(),
            "gan/epoch_pixel_loss": epoch_pixel_loss.result(),
            "gan/epoch_perceptual_loss": epoch_perceptual_loss.result(),
            "gan/epoch_cycle_loss": epoch_cycle_loss.result(),
            "gan/epoch_ssim": ssim_metric.result(),
            "gan/epoch_psnr": psnr_metric.result(),
            "gan/epoch_reconstruction_loss": reconstruction_metric.result(),
            "gan/epoch_completed": epoch + 1
        }
        
        if denoiser is not None:
            wandb_epoch_log.update({
                "gan/epoch_raw_latent_consistency_loss": epoch_raw_latent_loss.result(),
                "gan/epoch_denoiser_quality_loss": epoch_denoiser_quality_loss.result()
            })
        
        wandb.log(wandb_epoch_log)
        
        # Best model tracking and periodic saving
        current_ssim = ssim_metric.result().numpy()
        current_g_loss = g_loss_metric.result().numpy()
        
        # Combined score: Higher SSIM is better, Lower G_loss is better
        # Weight SSIM more heavily as we care about quality
        combined_score = current_ssim * 0.8 - current_g_loss * 0.2
        
        if combined_score < best_combined_score:
            best_combined_score = combined_score
            best_epoch = epoch + 1
            no_improvement_count = 0
            
            print(f"  NEW BEST MODEL! Combined Score: {combined_score:.4f} (SSIM: {current_ssim:.4f}, G_Loss: {current_g_loss:.4f})")
            
            # Save best models immediately
            print("  Saving best GAN models...")
            inverse_classifier.save_weights("inverse_classifier.weights.h5")
            decoder.save_weights("decoder.weights.h5")
            real_fake_classifier.save_weights("real_fake_classifier.weights.h5")
            
            wandb.log({
                "gan/best_combined_score": best_combined_score,
                "gan/best_epoch": best_epoch,
                "gan/best_ssim": current_ssim,
                "gan/best_g_loss": current_g_loss
            })
        else:
            no_improvement_count += 1
            print(f"  No improvement for {no_improvement_count} epochs (Best: Epoch {best_epoch}, Score: {best_combined_score:.4f})")
        
        # Periodic checkpoint saving (every N epochs regardless of performance)
        # if (epoch + 1) % save_frequency == 0:
        #     print(f"  Periodic checkpoint save at epoch {epoch + 1}...")
        #     inverse_classifier.save_weights(f"gan_checkpoint_epoch_{epoch+1}_inverse_classifier.weights.h5")
        #     decoder.save_weights(f"gan_checkpoint_epoch_{epoch+1}_decoder.weights.h5")
        #     real_fake_classifier.save_weights(f"gan_checkpoint_epoch_{epoch+1}_real_fake_classifier.weights.h5")
        
        # Early stopping check
        if no_improvement_count >= patience:
            print(f"  Early stopping triggered! No improvement for {patience} epochs.")
            print(f"  Best model was at epoch {best_epoch} with combined score {best_combined_score:.4f}")
            break
        
        # Comprehensive quality evaluation every few epochs
        if (epoch + 1) % 3 == 0:
            print(f"  Performing comprehensive quality evaluation...")
            quality_metrics = evaluate_reconstruction_quality(encoder, decoder, data, num_samples=200)
            
            print(f"  Comprehensive Quality Metrics:")
            print(f"    SSIM: {quality_metrics['ssim']:.4f}")
            print(f"    PSNR: {quality_metrics['psnr']:.2f}")
            print(f"    Reconstruction Loss: {quality_metrics['recon_loss']:.6f}")
            
            wandb.log({
                f"gan/comprehensive_ssim_epoch_{epoch+1}": quality_metrics['ssim'],
                f"gan/comprehensive_psnr_epoch_{epoch+1}": quality_metrics['psnr'],
                f"gan/comprehensive_recon_loss_epoch_{epoch+1}": quality_metrics['recon_loss']
            })
            
            log_reconstruction_images_to_wandb(encoder, decoder, data, f"gan_quality_epoch_{epoch+1}_")

        if (epoch + 1) % 2 == 0:
            log_generated_images_to_wandb(inverse_classifier, decoder, None, f"gan_epoch_{epoch+1}_")

    # Final quality evaluation
    print("Performing final quality evaluation...")
    final_quality = evaluate_reconstruction_quality(encoder, decoder, data, num_samples=500)
    
    print(f"Final Quality Metrics:")
    print(f"  SSIM: {final_quality['ssim']:.4f}")
    print(f"  PSNR: {final_quality['psnr']:.2f}")
    print(f"  Reconstruction Loss: {final_quality['recon_loss']:.6f}")
    
    wandb.log({
        "gan/final_ssim": final_quality['ssim'],
        "gan/final_psnr": final_quality['psnr'],
        "gan/final_reconstruction_loss": final_quality['recon_loss'],
        "gan/ssim_improvement": final_quality['ssim'] - initial_quality['ssim'],
        "gan/psnr_improvement": final_quality['psnr'] - initial_quality['psnr']
    })

    print("Enhanced GAN training completed!")
    
    # Load and save best models as final models
    print(f"\n Best model was found at epoch {best_epoch} with combined score {best_combined_score:.4f}")
    print("Loading best weights for final save...")
    
    try:
        # Load best weights
        inverse_classifier.load_weights("best_gan_inverse_classifier.weights.h5")
        decoder.load_weights("best_gan_decoder.weights.h5")
        real_fake_classifier.load_weights("best_gan_real_fake_classifier.weights.h5")
        print(" Best weights loaded successfully!")
        
        # Save as final models (overwrite any previous versions)
        inverse_classifier.save_weights("inverse_classifier.weights.h5")
        decoder.save_weights("decoder.weights.h5")
        real_fake_classifier.save_weights("real_fake_classifier.weights.h5")
        print(" Best models saved as final weights!")
        
    except Exception as e:
        print(f" Could not load best weights: {e}")
        print("Saving current (final epoch) weights instead...")
        inverse_classifier.save_weights("inverse_classifier.weights.h5")
        decoder.save_weights("decoder.weights.h5")
        real_fake_classifier.save_weights("real_fake_classifier.weights.h5")
    
    # Also keep a backup of the best weights with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        inverse_classifier.save_weights(f"best_gan_inverse_classifier_{timestamp}.weights.h5")
        decoder.save_weights(f"best_gan_decoder_{timestamp}.weights.h5")
        real_fake_classifier.save_weights(f"best_gan_real_fake_classifier_{timestamp}.weights.h5")
        print(f" Best models also saved with timestamp: {timestamp}")
    except:
        print(" Could not save timestamped backup")

if __name__ == "__main__":
    latent_dim = 256
    batch_size = 64 
    epochs = 3

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

    # Build models
    encoder = buildEncoder(latent_dim)
    decoder = buildDecoder(latent_dim)
    classifier = buildClassifier(latent_dim)
    inverse_classifier = buildInverseClassifier(latent_dim)
    denoiser = buildDenoiser(latent_dim)  # Add denoiser
    
    if os.path.exists("encoder.weights.h5") and os.path.exists("decoder.weights.h5"):
        try:
            print("Loading existing model weights...")
            encoder.load_weights("encoder.weights.h5")
            decoder.load_weights("decoder.weights.h5")
            
            # Load denoiser weights if available
            if os.path.exists("denoiser.weights.h5"):
                print("Loading existing denoiser weights...")
                denoiser.load_weights("denoiser.weights.h5")
            
            # Test if loaded weights work by doing a quick reconstruction test
            print("Testing loaded weights...")
            test_batch = next(iter(ds_train.take(1)))
            test_image = tf.cast(test_batch['image'][:1], tf.float32) / 255.0
            z_mean, z_log_var, z = encoder(test_image, training=False)
            test_recon = decoder(z, training=False)
            
            # Check if reconstruction is reasonable (not all zeros or garbage)
            recon_mean = tf.reduce_mean(test_recon).numpy()
            if recon_mean < 0.01 or recon_mean > 0.99:
                print(f"WARNING: Loaded weights seem corrupted (recon mean: {recon_mean}). Will retrain.")
                weights_ok = False
            else:
                print(f"Loaded weights seem OK (recon mean: {recon_mean})")
                weights_ok = True
        except Exception as e:
            print(f"Error loading weights: {e}. Will train from scratch.")
            weights_ok = False
    else:
        print("No existing model weights found. Training from scratch.")
        weights_ok = False

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
    
    # Capture denoiser summary
    denoiser_summary = []
    denoiser.summary()
    denoiser.summary(print_fn=lambda x: denoiser_summary.append(x))
    denoiser_summary_text = '\n'.join(denoiser_summary)
    
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
    
    # Log denoiser info
    wandb.log({
        "model_info/denoiser_params": denoiser.count_params(),
        "model_info/denoiser_trainable_params": sum([tf.size(w).numpy() for w in denoiser.trainable_weights]),
        "model_info/denoiser_layers": len(denoiser.layers)
    })
    
    # Log architecture summaries as text
    wandb.log({
        "model_summaries/encoder": wandb.Html(f"<pre>{encoder_summary_text}</pre>"),
        "model_summaries/decoder": wandb.Html(f"<pre>{decoder_summary_text}</pre>"),
        "model_summaries/denoiser": wandb.Html(f"<pre>{denoiser_summary_text}</pre>")
    })
    
    display_reconstructions(encoder, decoder, ds_train, num_images=1)
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
    generated_image = generate_image_from_features(feature_vector, inverse_classifier, decoder)  # No denoiser yet

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
    generated_image = generate_image_from_features(feature_vector, inverse_classifier, decoder)  # No denoiser yet

    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.clip(generated_image, 0, 1))
    plt.title(f"Strong: {', '.join(desired_features)}")
    plt.axis('off')

    plt.show()

    # Training with wandb logging
    print("Starting training with wandb logging...")
    
    # Train VAE first - this is CRITICAL!
    if not weights_ok:
        print("Training VAE from scratch...")
        train(ds_train, encoder, decoder, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
    else:
        print("Skipping VAE training - using existing weights")
    
    # Train denoiser after VAE
    print("\n=== Training Denoiser ===")
    denoiser_epochs = max(1, epochs // 2)  # Use fewer epochs for denoiser
    train_denoiser(ds_train, encoder, decoder, denoiser, epochs=denoiser_epochs, batch_size=batch_size, latent_dim=latent_dim)
    
    # Log sample reconstructions after VAE training
    log_generated_images_to_wandb(inverse_classifier, decoder, denoiser, "post_vae_")

    log_reconstruction_images_to_wandb(encoder, decoder, ds_train, "post_vae_")
    
    # Train inverse classifier
    train_inverse_classifier(ds_train, encoder, inverse_classifier, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
    
    # Log sample generations after inverse classifier training
    log_generated_images_to_wandb(inverse_classifier, decoder, denoiser, "post_inverse_")
    
    # Train classifier
    # train_classifier(ds_train, encoder, classifier, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
    
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
    # Use the batch size calculated in train_gan function (batch_size // 2)
    train_gan(ds_train, inverse_classifier, decoder, real_fake_classifier, encoder, denoiser, epochs=gan_epochs, batch_size=batch_size, latent_dim=latent_dim)
    
    # Log post-GAN generations
    log_generated_images_to_wandb(inverse_classifier, decoder, denoiser, "post_gan_")

    print("\n=== Generating Images from Features ===")
    
    desired_features = ['Young', 'Smiling', 'Blond_Hair', 'Attractive']
    intensities = [0.8, 0.9, 0.7, 0.8]
    feature_vector = create_feature_vector_from_descriptions(desired_features, intensities=intensities)
    
    generated_image = generate_image_from_features(feature_vector, inverse_classifier, decoder, denoiser)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.clip(generated_image, 0, 1))
    plt.title(f"Strong: {', '.join(desired_features)}")
    plt.axis('off')
    
    # Subtle version
    intensities_subtle = [0.3, 0.4, 0.2, 0.3]
    feature_vector_subtle = create_feature_vector_from_descriptions(desired_features, intensities=intensities_subtle)
    generated_image_subtle = generate_image_from_features(feature_vector_subtle, inverse_classifier, decoder, denoiser)
    
    plt.subplot(2, 3, 2)
    plt.imshow(np.clip(generated_image_subtle, 0, 1))
    plt.title(f"Subtle: {', '.join(desired_features)}")
    plt.axis('off')
    
    # Male with varying intensities
    desired_features2 = ['Male', 'Eyeglasses', 'Mustache', 'Young']
    intensities2 = [0.9, 0.8, 0.6, 0.7]
    feature_vector2 = create_feature_vector_from_descriptions(desired_features2, intensities=intensities2)
    generated_image2 = generate_image_from_features(feature_vector2, inverse_classifier, decoder, denoiser)
    
    plt.subplot(2, 3, 3)
    plt.imshow(np.clip(generated_image2, 0, 1))
    plt.title(f"Male Strong")
    plt.axis('off')
    
    # Progressive smile intensity
    for i, smile_intensity in enumerate([0.2, 0.5, 0.8]):
        features = ['Young', 'Smiling', 'Attractive']
        intensities = [0.8, smile_intensity, 0.7]
        fv = create_feature_vector_from_descriptions(features, intensities=intensities)
        img = generate_image_from_features(fv, inverse_classifier, decoder, denoiser)
        
        plt.subplot(2, 3, 4 + i)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"Smile: {smile_intensity}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display some reconstructions
    display_reconstructions(encoder, decoder, ds_train, num_images=1)
    
    # Log final results
    log_generated_images_to_wandb(inverse_classifier, decoder, denoiser, "final_")
    
    # Finish wandb run
    wandb.finish()