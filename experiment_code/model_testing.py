#%%
from sklearn.model_selection import train_test_split
import json
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from imageio import imread
import diplib as dip
import os
import matplotlib.pyplot as plt

plot = False

def create_test_data(n_objects, object_projection_dir, target_projection_dir, random_seed = 4):
    X_test = list()
    y_test = list()
    for n in range(n_objects):
        for p in range(180):
            obj = imread(os.path.join(object_projection_dir, f"object_{n}", f"projection_{p}.tif")).astype(float)
            obj /= 65535  # NOTE: uint16
            img_max = np.max(obj)
            img_min = np.min(obj)
            normalized = (obj - img_min) / (img_max - img_min)
            X_test.append(np.asarray(dip.ColorSpaceManager.Convert(dip.Image(normalized), 'RGB')))

            target = imread(os.path.join(target_projection_dir, f"targets_{n}", f"projection_{p}.tif")).astype(float)
            target = dip.Image(target)
            target = dip.IsodataThreshold(target)
            y_test.append(np.asarray(target).astype(int))


    X_test = np.array(X_test); X_test.reshape(X_test.shape)
    y_test = np.array(y_test); y_test.reshape(y_test.shape)

    return X_test, y_test

#%%

X, y = create_test_data(25, "./data/test_object_projections/25_objects", "./data/test_target_projections/25_objects")

abspath = "./data/saved_models"
saved_models_path = os.listdir(abspath)
saved_models = sorted(saved_models_path, key=lambda x: int(x.split("_")[0]))

#%%

obj = list()
val = list()
std_val = list()

histories = list()

for model_path in saved_models:
    n_obj = int(model_path.split("_")[0])
    path = os.path.join(abspath, model_path)

    history = json.load(open(os.path.join(path, f"history_{n_obj}_objects.json")))
    model = tf.keras.models.load_model(os.path.join(path, f"unet_{n_obj}_objects.keras"))

    histories.append(history)

    print(f"!!! --- {n_obj} objects --- !!!")
    predicted = model.predict(X)
    predictions = np.argmax(predicted, axis=3)

    measurements = list()
    for idx, p in enumerate(predicted):
        confusion_matrix = tf.math.confusion_matrix(y[idx].reshape(-1), predictions[idx].reshape(-1))
        TP_bg = int(confusion_matrix[0][0])
        TP_obj = int(confusion_matrix[1][1])
        FP_bg = int(confusion_matrix[1][0])
        FP_obj = int(confusion_matrix[0][1])

        image_based_average_class_accuracy = .5 * ((TP_obj / (TP_obj + FP_bg)) + (TP_bg / (TP_bg + FP_obj)))
        measurements.append(image_based_average_class_accuracy)

    if plot:
        fig, arr = plt.subplots(1, 3, figsize=(15,15))
        arr[0].imshow(X[-1])
        arr[1].imshow(y[-1])
        arr[2].imshow(predictions[-1])
    
    obj.append(n_obj)
    val.append(np.mean(measurements))
    std_val.append(np.std(measurements))

#%%
plt.plot(obj, val, label = f"Average class accuracy")
plt.fill_between(obj, np.subtract(val, std_val), np.add(val, std_val), alpha=0.2)
plt.title("Average class accuracies for different number of training objects")
plt.xlabel("Amount of training objects")
plt.ylabel("Average class accuracy")
plt.show()

#%%
epochs = 20
for idx, history in enumerate(histories):
    n_objects = obj[idx]
    x = list(range(epochs))
    y = history["accuracy"]
    plt.plot(x, y, label=f"{n_objects} objects")

plt.legend(fontsize=5)
plt.title("Accuracy per epoch")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

#%%
epochs = 20
for idx, history in enumerate(histories):
    n_objects = obj[idx]
    x = list(range(epochs))
    y = history["val_accuracy"]
    plt.plot(x, y, label=f"{n_objects} objects")

plt.legend(fontsize=5)
plt.title("Validation set accuracy per epoch")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

#%%

long_model = tf.keras.models.load_model("./data/saved_models/long/5_objects/unet_5_objects.keras")
long_history = json.load(open("./data/saved_models/long/5_objects/history_5_objects.json"))

#%%

n_objects = 5
x = list(range(200))
y = long_history["accuracy"]
plt.plot(x, y, label="Training set")
y = long_history["val_accuracy"]
plt.plot(x, y, label=f"Validation set")

plt.legend()
plt.title("Accuracy per epoch")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

#%%

X, y = create_test_data(25, "./data/test_object_projections/25_objects", "./data/test_target_projections/25_objects")

predicted = long_model.predict(X)
predictions = np.argmax(predicted, axis=3)

measurements = list()
for idx, p in enumerate(predicted):
    confusion_matrix = tf.math.confusion_matrix(y[idx].reshape(-1), predictions[idx].reshape(-1))
    TP_bg = int(confusion_matrix[0][0])
    TP_obj = int(confusion_matrix[1][1])
    FP_bg = int(confusion_matrix[1][0])
    FP_obj = int(confusion_matrix[0][1])

    image_based_average_class_accuracy = .5 * ((TP_obj / (TP_obj + FP_bg)) + (TP_bg / (TP_bg + FP_obj)))
    measurements.append(image_based_average_class_accuracy)

print(np.mean(measurements))
print(np.std(measurements))
