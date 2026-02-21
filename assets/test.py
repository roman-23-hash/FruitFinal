import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def run_guava_binary_test(img_path):
    # 1. LOAD MODEL
    model_file = "E:\project\Guava (Psidium guajava) fruit digital and thermal Images\guava_perfect_expert1.h5"
    if not os.path.exists(model_file):
        print("❌ Error: Model 'guava_perfect_expert.h5' not found.")
        return
    model = tf.keras.models.load_model(model_file)
    
    # 2. LOAD RGB IMAGE
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("❌ Error: Image not found.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. COLOR SPECTRUM CHECK (Guava: Green to Yellow)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Define range: Hue 25-85 (Green-Yellow spectrum)
    lower_guava = np.array([25, 40, 40])
    upper_guava = np.array([85, 255, 255])
    
    guava_mask = cv2.inRange(hsv, lower_guava, upper_guava)
    match_pct = (np.count_nonzero(guava_mask) / (img_bgr.shape[0] * img_bgr.shape[1])) * 100

    # 4. BINARY DECISION GATE
    # If the fruit isn't in the guava color spectrum, it's 'Not Guava'
    if match_pct < 20.0:
        print("RESULT: Not Guava")
        print("⚠️ Processing stopped: Input does not match guava characteristics.")
        return # Hard exit: No thermal map or prediction is generated

    # 5. MODEL PREDICTION (Only for verified Guavas)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_res = cv2.resize(img_gray, (512, 512))
    img_input = img_res.reshape(1, 512, 512, 1).astype('float32') / 255.0

    thermal_p, ripeness_p, _ = model.predict(img_input)

    # 6. SUCCESS OUTPUT
    rip_score = ripeness_p[0][0] * 100
    thermal_map = thermal_p[0].reshape(512, 512)

    print("-" * 25)
    print("RESULT: Guava")
    print(f"RIPENESS: {rip_score:.2f}%")
    print("-" * 25)

    # 7. VISUALIZATION
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Input Fruit (RGB)")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Thermal Map (Guava Ripeness: {rip_score:.1f}%)")
    plt.imshow(thermal_map, cmap='magma')
    plt.colorbar(label='Heat Intensity')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test path: Red apples will trigger 'Not Guava' and stop
    run_guava_binary_test(r'C:\Users\PAVILION\Pictures\Screenshots\Screenshot 2026-02-18 144113.png')