import tensorflow as tf
import data.datasetUtils as customUtil
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import numpy as np
import tensorflow_datasets as tfds
print("TensorFlow version:", tf.__version__)


if __name__ == "__main__":
    dataset = customUtil.create_dataset('data/formData.csv', 2)  # note, will not work if we convert to tfrecord datatype
    dataset_len = dataset.cardinality().numpy()

    train_size = int(0.5 * dataset_len)
    test_size = int(0.5 * dataset_len)


    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    # train_x = np.concatenate([x for x, y in train_dataset], axis=0)
    # train_y = np.concatenate([y for x, y in train_dataset], axis=0)

    # print(train_y)

    # train_data, train_labels = tuple(zip(*train_dataset))
    # test_data, test_labels = tuple(zip(*test_dataset))

    # train_data = np.asarray(train_data)
    # train_labels = np.asarray(train_labels)
    # test_data = np.asarray(test_data)
    # test_labels = np.asarray(test_labels)

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    # plt.figure(figsize=(10,10))
    # for i, (x, y) in enumerate(train_dataset.take(25)):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(x)
    #     # The CIFAR labels happen to be arrays, 
    #     # which is why you need the extra index
    #     plt.xlabel(y.numpy())
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

    model.summary()


    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    history = model.fit(train_dataset, epochs=10, 
                    validation_data=test_dataset)
    print(history.history.keys())
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

    print(test_acc)


    tf.saved_model.save(model, './savedModel/1/')








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
