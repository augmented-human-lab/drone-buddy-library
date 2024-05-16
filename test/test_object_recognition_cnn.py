import tensorflow as tf


def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        # Adjust input_shape based on your image size
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # num_classes should be the number of your categories
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Example usage
num_classes = 3  # Change this to the number of your categories
model = build_model(num_classes)
model.summary()

train_dir = r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectrecognition\resources\model\data\training_data'  # Adjust to the path where your training data is stored
validation_dir = r'C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectrecognition\resources\model\data\validation_data'  # Adjust to the path where your validation data is stored

# Assuming you have your data in 'train' and 'validation' directories
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Make sure this matches the input_shape in your model
    batch_size=32,
    class_mode='categorical')

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

steps_per_epoch = len(train_generator)  # This automatically calculates the correct number of steps per epoch

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,  # Adjust based on the size of your validation data
    verbose=1  # Set verbose to 1 or 2 for more detailed logging
)

# Assuming 'train_generator' is available and was used for training
class_indices = train_generator.class_indices
# This gives you a dictionary mapping class names to class indices

# To get a mapping from indices to class names
class_names = {v: k for k, v in class_indices.items()}
print(class_names)
# Save the model
model_save_path = r'C:\Users\Public\projects\drone-buddy-library\test\model\cnn\model.keras'
model.save(model_save_path)

