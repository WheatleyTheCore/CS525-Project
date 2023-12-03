import tensorflow as tf
import data.datasetUtils as customUtil
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import numpy as np
import tensorflow_datasets as tfds
import time 
print("TensorFlow version:", tf.__version__)


if __name__ == "__main__":
    dataset = customUtil.create_dataset('data/formData.csv', 2)  # note, will not work if we convert to tfrecord datatype
    dataset_len = dataset.cardinality().numpy()

    train_size = int(0.5 * dataset_len)
    test_size = int(0.5 * dataset_len)


    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    output_decoder = {
        0: 'Sitting',
        1: 'Standing',
        2: "Walking",
        3: "Crouching"
    }


    # plt.figure(figsize=(10,10))
    # for i, (x, y) in enumerate(train_dataset.take(10)):
    #     plt.subplot(2,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(x[0])
    #     plt.xlabel(output_decoder[y[0].numpy()])
    # plt.show()

    model = models.Sequential()
    model.add(layers.Conv2D(64, (30, 1), activation='relu', input_shape=(40, 6, 1)))
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Conv2D(64, (3, 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 1)))
    # model.add(layers.Conv2D(64, (2, 1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(4))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    history = model.fit(train_dataset, epochs=10, 
                    validation_data=test_dataset)
    # print(history.history.keys())
    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.0, 1])
    # plt.legend(loc='lower right')
    # plt.show()



    # Create some different quantization options
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dynamic_quant_model = converter.convert()

    converter.target_spec.supported_types = [tf.float16]
    converter.representative_dataset = test_dataset
    float16_model = converter.convert()

    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_types = [tf.int8]
    # converter.inference_input_type = tf.int8  # or tf.uint8
    # converter.inference_output_type = tf.int8  # or tf.uint8
    # int8_model = converter.convert()

    
    models = ['float16', 'dynamic', 'full']
    times = []
    accuracies = []

    # tf.config.experimental.reset_memory_stats()
    # start_time = time()
    # test_loss, test_acc = int8_model.evaluate(test_dataset, verbose=2)
    # times.append(time() - start_time)
    # accuracies.append(test_acc)
    # peak_mem.append(tf.config.experimental.get_memory_info("CPU").peak)

    start_time = time.time()
    test_loss, test_acc = float16_model.evaluate(test_dataset, verbose=2)
    times.append(time.time() - start_time)
    accuracies.append(test_acc)

    start_time = time.time()
    test_loss, test_acc = dynamic_quant_model.evaluate(test_dataset, verbose=2)
    times.append(time.time() - start_time)
    accuracies.append(test_acc)

    start_time = time.time()
    test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
    times.append(time.time() - start_time)
    accuracies.append(test_acc)

    plt.plot(models, times, label='times')
    plt.plot(models, accuracies, label='accuracy')
    plt.xlabel('Model')
    plt.legend(loc='lower right')
    plt.show()




    #tf.saved_model.save(model, './savedModel/1/')








    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10)
    # ])

    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # model.compile(optimizer='adam',
    #             loss=loss_fn,
    #             metrics=['accuracy'])

    # model.fit(x_train, y_train, epochs=5)

    # model.evaluate(x_test,  y_test, verbose=2)


    # tf.saved_model.save(model, './savedModel/1/')
