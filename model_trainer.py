import tensorflow as tf

from globals import num_classes, num_epochs, train_images, val_images

if __name__ == '__main__':

    # modeling
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg',
    )

    pretrained_model.trainable = False

    inputs = pretrained_model.input
    # classification layers
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x) # number of classes

    model = tf.keras.Model(inputs, outputs)

    # training
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=num_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )

    model.save('./model');
    
