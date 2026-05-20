from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent

model = tf.keras.models.load_model(BASE_DIR / "models" / "cnn_model.h5")

test_dir = BASE_DIR / "data" / "test"

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)

print(confusion_matrix(test_generator.classes, predicted_classes))
print(classification_report(test_generator.classes, predicted_classes))