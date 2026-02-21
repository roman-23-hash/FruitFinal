import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import mixed_precision

# --- 1. HARDWARE & MEMORY OPTIMIZATION ---
# RTX 3050 Laptop GPUs need mixed precision to handle 512px resolution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # float16 math reduces VRAM usage by 50%
        mixed_precision.set_global_policy('mixed_float16') 
        print("✅ GPU Configured: Mixed Precision Active")
    except Exception as e:
        print(f"GPU Error: {e}")

# --- 2. PATHS & CONFIG ---
IMG_SIZE = 512 
EPOCHS = 60
digital_path = r'E:\project\Guava (Psidium guajava) fruit digital and thermal Images\DIGITAL PHOTOS'
thermal_path = r'E:\project\Guava (Psidium guajava) fruit digital and thermal Images\THERMAL IMAGES'

# --- 3. THE "EXPERT" DATA GENERATOR ---
class GuavaDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, d_files, t_files, augment=True):
        self.d_files = d_files
        self.t_files = t_files
        self.augment = augment

    def __len__(self):
        return len(self.d_files)

    def __getitem__(self, idx):
        # Read as Grayscale to emphasize texture over color noise
        img_d = cv2.imread(self.d_files[idx], 0)
        img_t = cv2.imread(self.t_files[idx], 0)
        img_d = cv2.resize(img_d, (IMG_SIZE, IMG_SIZE))
        img_t = cv2.resize(img_t, (IMG_SIZE, IMG_SIZE))

        # Strategic Augmentation for "Unseen" Accuracy
        if self.augment and np.random.random() > 0.5:
            flip = np.random.choice([-1, 0, 1])
            img_d = cv2.flip(img_d, flip)
            img_t = cv2.flip(img_t, flip)

        X = np.expand_dims(img_d, axis=(0, -1)).astype('float32') / 255.0
        Y_thermal = np.expand_dims(img_t, axis=(0, -1)).astype('float32') / 255.0
        
        # Labeling Logic (Assumes filenames contain ripeness state)
        name = self.d_files[idx].lower()
        if 'guava' in name:
            is_guava = 1.0
            # Higher precision mapping
            rip = 0.10 if 'immature' in name else 0.90 if 'mature' in name else 0.50
        else:
            is_guava = 0.0
            rip = 0.0
        
        return X, {
            'thermal_out': Y_thermal, 
            'ripeness_out': np.array([[rip]], dtype='float32'),
            'guava_guard': np.array([[is_guava]], dtype='float32')
        }

# --- 4. THE MULTI-TASK U-NET ARCHITECTURE ---
def build_expert_model():
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1])

    # Encoder: Extracting Texture and Shape
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1) # 256x256

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2) # 128x128

    # --- Head 1: Guava Guard (Classification) ---
    guard = layers.GlobalAveragePooling2D()(p2)
    guava_guard = layers.Dense(1, activation='sigmoid', name='guava_guard', dtype='float32')(guard)

    # --- Head 2: Thermal Generator (U-Net Decoder with Skip Connections) ---
    # These skip connections ensure the thermal image looks sharp, not blurry
    u1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(p2)
    u1 = layers.concatenate([u1, c2]) 
    u1 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)

    u2 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(u1)
    u2 = layers.concatenate([u2, c1]) 
    thermal_out = layers.Conv2D(1, 3, activation='sigmoid', padding='same', name='thermal_out', dtype='float32')(u2)

    # --- Head 3: Ripeness Expert (High-Precision Regression) ---
    # Using GlobalMaxPooling helps the model find the "ripest" spots on the fruit
    r = layers.GlobalMaxPooling2D()(p2)
    r = layers.Dense(128, activation='relu')(r)
    r = layers.Dropout(0.2)(r)
    ripeness_out = layers.Dense(1, activation='sigmoid', name='ripeness_out', dtype='float32')(r)

    return models.Model(inputs, [thermal_out, ripeness_out, guava_guard])

# --- 5. TRAINING ---
if __name__ == '__main__':
    def get_files(path):
        return sorted([os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    d_list = get_files(digital_path)
    t_list = get_files(thermal_path)
    
    train_gen = GuavaDataGenerator(d_list, t_list)

    model = build_expert_model()
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss={
            'thermal_out': 'mae', 
            'ripeness_out': 'huber', # Huber is more robust for accuracy than MSE
            'guava_guard': 'binary_crossentropy'
        },
        # Prioritize Ripeness and Guard over the visual look of the thermal image
        loss_weights={'thermal_out': 2.0, 'ripeness_out': 900.0, 'guava_guard': 300.0}
    )

    model.fit(train_gen, epochs=EPOCHS)
    model.save("guava_perfect_expert1.h5")