# train.py
import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --------- CONFIG ----------
DATA_DIR = 'dataset'        # should contain train/ and val/ folders
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
LR = 1e-4
BEST_MODEL = 'best_model.h5'
FINAL_MODEL = 'waste_model.h5'
CLASS_INDICES_FILE = 'class_indices.json'
IDX_TO_CLASS_FILE = 'idx_to_class.json'
PLOT_FILE = 'training_history.png'
# --------------------------

train_dir = os.path.join(DATA_DIR, 'train')
val_dir = os.path.join(DATA_DIR, 'val')

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# Save class indices
with open(CLASS_INDICES_FILE, 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)
print("Saved class indices to", CLASS_INDICES_FILE)
# create idx->class mapping (string keys for JSON)
idx_to_class = {str(v): k for k, v in train_gen.class_indices.items()}
with open(IDX_TO_CLASS_FILE, 'w') as f:
    json.dump(idx_to_class, f, indent=2)
print("Saved idx->class mapping to", IDX_TO_CLASS_FILE)

num_classes = len(train_gen.class_indices)
print("Num classes:", num_classes)

# Build model (transfer learning)
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
preds = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=preds)
model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint(BEST_MODEL, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_accuracy', patience=4, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# Train
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# Optionally fine-tune some layers
print("Unfreezing top layers and fine-tuning...")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(LR/10), loss='categorical_crossentropy', metrics=['accuracy'])
history_f = model.fit(train_gen, epochs=5, validation_data=val_gen, callbacks=callbacks)

# Save final model
model.save(FINAL_MODEL)
print("Saved final model to", FINAL_MODEL)

# Plot training history
def plot_hist(h1, h2=None, fname=PLOT_FILE):
    acc = h1.history.get('accuracy', [])
    val_acc = h1.history.get('val_accuracy', [])
    loss = h1.history.get('loss', [])
    val_loss = h1.history.get('val_loss', [])
    if h2:
        acc += h2.history.get('accuracy', [])
        val_acc += h2.history.get('val_accuracy', [])
        loss += h2.history.get('loss', [])
        val_loss += h2.history.get('val_loss', [])
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.tight_layout()
    plt.savefig(fname)
    print("Saved training plot to", fname)

plot_hist(history, history_f if 'history_f' in globals() else None)
