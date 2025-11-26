"""
eval_and_gradcam.py
Load saved model (.h5), run on images/test folder, and produce Grad-CAM visualizations.

Usage:
    python eval_and_gradcam.py --model path/to/model.h5 --image path/to/image.jpg
    python eval_and_gradcam.py --model path/to/model.h5 --dir path/to/images_folder
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

def load_img(path, img_size=(224,224)):
    img = Image.open(path).convert("RGB").resize(img_size)
    arr = np.array(img).astype(np.float32)
    return arr

def get_model_and_preprocess(model_path, backbone="ResNet50"):
    model = tf.keras.models.load_model(model_path, compile=False)
    # infer preprocess from model name (user must ensure consistency)
    if "resnet" in model_path.lower() or backbone=="ResNet50":
        preprocess = tf.keras.applications.resnet.preprocess_input
    else:
        preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
    return model, preprocess

def grad_cam(model, img_array, class_index=None, last_conv_layer_name=None, preprocess=None):
    # img_array: HxWx3, float
    img_tensor = np.expand_dims(img_array, axis=0)
    if preprocess is not None:
        img_tensor = preprocess(img_tensor)
    # find last conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError("Could not find a Conv2D layer in the model to use for Grad-CAM.")
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if class_index is None:
            class_index = np.argmax(predictions[0])
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (tf.math.reduce_max(heatmap) + 1e-9)
    heatmap = heatmap.numpy()
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=-1)  # HxWx3
    heatmap = Image.fromarray(heatmap).resize((img_array.shape[1], img_array.shape[0]))
    return np.array(heatmap), class_index

def overlay_heatmap(img, heatmap, alpha=0.4, cmap='jet'):
    import cv2
    heatmap_color = cv2.applyColorMap(heatmap[:,:,0], cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = img * (1 - alpha) + heatmap_color * alpha
    overlay = np.uint8(overlay)
    return overlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--dir", default=None)
    parser.add_argument("--backbone", default="ResNet50")
    args = parser.parse_args()

    model, preprocess = get_model_and_preprocess(args.model, backbone=args.backbone)
    if args.image:
        img_arr = load_img(args.image)
        heatmap, cls = grad_cam(model, img_arr, preprocess=preprocess, last_conv_layer_name=None)
        overlay = overlay_heatmap(img_arr, heatmap)
        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1); plt.imshow(img_arr.astype(np.uint8)); plt.title("Original"); plt.axis("off")
        plt.subplot(1,3,2); plt.imshow(heatmap); plt.title("Grad-CAM"); plt.axis("off")
        plt.subplot(1,3,3); plt.imshow(overlay); plt.title(f"Overlay (pred class {cls})"); plt.axis("off")
        plt.tight_layout()
        plt.savefig("gradcam_output.png", bbox_inches='tight')
        plt.show()
    elif args.dir:
        out_dir = "gradcam_batch"
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(args.dir):
            if not fname.lower().endswith((".jpg",".png",".jpeg")): continue
            path = os.path.join(args.dir, fname)
            img_arr = load_img(path)
            heatmap, cls = grad_cam(model, img_arr, preprocess=preprocess, last_conv_layer_name=None)
            overlay = overlay_heatmap(img_arr, heatmap)
            outp = np.hstack([img_arr.astype(np.uint8), heatmap, overlay])
            Image.fromarray(outp).save(os.path.join(out_dir, fname))
        print("Saved Grad-CAM images to", out_dir)
    else:
        print("Provide --image or --dir. Example:\n python eval_and_gradcam.py --model plant_disease_model.h5 --image sample.jpg")

if __name__ == "__main__":
    main()
