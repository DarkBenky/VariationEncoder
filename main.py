import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Input
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

# check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)


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
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(data):
            images = tf.cast(batch['image'], tf.float32) / 255.0
            
            # Convert nested attributes dict to tensor array
            attr_list = []
            for attr_name in sorted(batch['attributes'].keys()):  # Sort for consistent order
                attr_list.append(tf.cast(batch['attributes'][attr_name], tf.float32))
            attributes = tf.stack(attr_list, axis=1)  # Shape: (batch_size, 40)
            
            total_loss, loss = train_step(images, attributes)
            if step % 100 == 0:
                print(f"Step {step}, Total Loss: {total_loss.numpy():.6f}, BCE Loss: {loss.numpy():.6f}")
        # Save model checkpoints
        classifier.save_weights(f"classifier.weights.h5")
    print("Classifier training completed.")

def buildDecoder(latent_dim):
    decoder_input = Input(shape=(latent_dim,), name="decoder_input")
    x = Dense(8 * 8 * 512)(decoder_input)
    x = Reshape((8, 8, 512))(x)
    x = Conv2DTranspose(256, 4, strides=2, padding="same")(x)  # 16x16
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(128, 4, strides=2, padding="same")(x)  # 32x32
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(64, 4, strides=2, padding="same")(x)   # 64x64
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(32, 4, strides=2, padding="same")(x)   # 128x128
    x = tf.keras.layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    decoder_output = Conv2DTranspose(3, 4, strides=2, activation="sigmoid", padding="same", name="decoder_output")(x)  # 256x256
    decoder = Model(decoder_input, decoder_output, name="decoder")
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
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
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
            step_count += 1
            global_step.assign_add(1)
            
            if step % 100 == 0:
                print(f"Step {step}, Total: {loss.numpy():.4f}, Recon: {recon_loss.numpy():.4f}, KL: {kl_loss.numpy():.4f}")
        
        avg_loss = epoch_loss / step_count
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        
        # Save model checkpoints
        encoder.save_weights(f"encoder.weights.h5")
        decoder.save_weights(f"decoder.weights.h5")
    
    print("Training completed.")

def buildInverseClassifier(latent_dim):
    """Network that maps from attributes to latent space"""
    inverse_input = Input(shape=(40,), name="inverse_input")  # 40 CelebA attributes
    x = Dense(256, activation="relu")(inverse_input)
    x = Dense(512, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    inverse_output = Dense(latent_dim, name="inverse_output")(x)
    inverse_classifier = Model(inverse_input, inverse_output, name="inverse_classifier")
    return inverse_classifier

def train_inverse_classifier(data, encoder, inverse_classifier, epochs=10, batch_size=32, latent_dim=128):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z_real = encoder(x, training=False)
            z_pred = inverse_classifier(y, training=True)
            
            # Use z_mean for more stable training
            loss = mse_loss_fn(z_mean, z_pred)
            
            # Add L2 regularization
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in inverse_classifier.trainable_weights])
            total_loss = loss + 0.0001 * l2_loss
            
        grads = tape.gradient(total_loss, inverse_classifier.trainable_weights)
        optimizer.apply_gradients(zip(grads, inverse_classifier.trainable_weights))
        return total_loss

    for epoch in range(epochs):
        print(f"Inverse Classifier Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(data):
            images = tf.cast(batch['image'], tf.float32) / 255.0
            
            # Convert nested attributes dict to tensor array
            attr_list = []
            for attr_name in sorted(batch['attributes'].keys()):
                attr_list.append(tf.cast(batch['attributes'][attr_name], tf.float32))
            attributes = tf.stack(attr_list, axis=1)
            
            total_loss = train_step(images, attributes)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {total_loss.numpy():.6f}")
        
        # Save model checkpoints
        inverse_classifier.save_weights(f"inverse_classifier.weights.h5")
    print("Inverse classifier training completed.")

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

if __name__ == "__main__":
    latent_dim = 256  # Increased latent dimension
    batch_size = 64   # Reduced batch size for better gradients
    epochs = 5       # More epochs

    desired_features = ['Young', 'Smiling', 'Blond_Hair', 'Attractive']
    intensities = [0.8, 0.9, 0.7, 0.8]
    feature_vector = create_feature_vector_from_descriptions(desired_features)

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

    display_reconstructions(encoder, decoder, ds_train, num_images=1)
    train(ds_train, encoder, decoder, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)

    classifier = buildClassifier(latent_dim)
    if os.path.exists("classifier.weights.h5"):
        print("Loading existing classifier weights...")
        classifier.load_weights("classifier.weights.h5")

    train_classifier(ds_train, encoder, classifier, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
   
    # Build and train inverse classifier
    inverse_classifier = buildInverseClassifier(latent_dim)
    if os.path.exists("inverse_classifier.weights.h5"):
        print("Loading existing inverse classifier weights...")
        inverse_classifier.load_weights("inverse_classifier.weights.h5")
    
    train_inverse_classifier(ds_train, encoder, inverse_classifier, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
    

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