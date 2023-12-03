import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from collections import deque

'''
So here's the plan. We're gonna create a whole bunch of 2D data matricies that are 
gonna have the columns be the sensor inputs (e.g. column one is accel x, column
2 is accel y, etc) and the rows be time. We'll just slide down form data with some
specified stride to create a whole bunch of 40-entry (2 seconds of data) long tensors,
of which the data will all be of the same label. (e.g. we'll slide down our data entires
until we hit something that's different.) Then, we'll just convert that to a dataset
and save it. 
'''



def create_dataset(data_file, stride=20):


    data_buffer = deque([], maxlen=40)  # a circular buffer-esque thing for our data recording. 40 entries will give us 2 seconds of data.
    chunked_data_tensors = []           # for storing our tensors and whatnot
    chunked_data_labels = []

    output_encoder = {
        'Sitting': 0,
        'Standing': 1,
        "Walking": 2,
        "Crouching": 3
    }

    line_count = 0


    with open(data_file, mode ='r') as file:
        csvFile = csv.reader(file)
        previous_label = None
        for row in csvFile:
            if line_count == 0:
                line_count += 1
                continue                     # skip headers
                
            if row[6] != previous_label and previous_label != None:
                chunked_data_tensors.append(np.array(list(data_buffer), dtype=np.float16)) # if we've hit another label, just record what we have. There will be some overlap, but that's fine I think
                chunked_data_labels.append(output_encoder[previous_label])
                #print(chunked_data_tensors)

            data_buffer.append([row[0], row[1], row[2], row[3], row[4], row[5]]) # use small dtype for efficiency!
            line_count += 1

            if line_count % stride == 0 and line_count > 40:
                chunked_data_tensors.append(np.array(list(data_buffer), dtype=np.float16)) # if we've hit another label, just record what we have. There will be some overlap, but that's fine I think
                chunked_data_labels.append(output_encoder[row[6]])

            previous_label = row[6]

    return tf.data.Dataset.from_tensor_slices((chunked_data_tensors, chunked_data_labels)).shuffle(line_count+30).batch(10)

    


if __name__ == "__main__":
    dataset = create_dataset('formData.csv')
    print('printing dataset')

    for x, y in dataset.take(1):
        print(x, y)