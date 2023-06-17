from pandas.io.xml import preprocess_data
from pycaret.anomaly import predict_model, save_model
import time

def iid_partition(data, num_clients):
    """
    Partitions the data into num_clients clients, with IID data.

    Args:
        data: Input dataset to be partitioned.
        num_clients: Number of clients to partition the data into.

    Returns:
        A list of num_clients tuples, where each tuple contains the partitioned data for a particular client.
    """
    data_size = len(data)
    # get the number of elements in the variable data
    data_indices = np.arange(data_size)
    np.random.shuffle(data_indices)
    #random order during model training

    # Calculate the size of each partition
    partition_size = int(data_size / num_clients)

    # Partition the data and store in a list
    partitions = []
    for i in range(num_clients):
        partition_start = i * partition_size
        partition_end = (i + 1) * partition_size
        partition_indices = data_indices[partition_start:partition_end]
        partition = data.iloc[partition_indices]
        partitions.append(partition)

    return partitions

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pycaret.regression import compare_models
from pycaret.classification import *
import psutil

# Load and preprocess the data
data = pd.read_csv('houseprice11.csv')
data = data.drop(["date","air quality level", "region"],axis=1)

data.dropna(inplace=True)
data = preprocess_data(data)

# start recording time and CPU utilization
# start_time = time.time()
# start_cpu_utilization = psutil.cpu_percent()

# Divide the data into 4 parts with IID
partitions = iid_partition(data, 4)


# Train models for each partition sequentially
for i in range(len(partitions)):
    if i == 0:
        partition_train = partitions[i]
    else:
        partition_train = pd.concat(partitions[:i], ignore_index=True)
    partition_test = partitions[i]

    # Train the model using PyCaret's compare_models function
    clf = setup(data=partition_train, target='class')
    model = create_model('et')

    # 预测输出
    # if i==3:
    #     print("predict")
    #     new_data = pd.read_csv('Test_Data.csv')
    #     new_data = new_data.drop(["date", "air quality level", "region"], axis=1)
    #
    #     predictions = predict_model(model, data=new_data)
    #     # predictions.to_csv('predictions.csv', index=False, header=True, columns=['class'])
    #     predictions.to_csv('predictions.csv')
    #     print(predictions.iloc[:, -2])


    # model = create_model('et')
    # tuned_et = tune_model(model)
    # evaluate the tuned Extra Trees Classifier model
    # evaluate_model(tuned_et)

    # et = create_model('et', fold=5)
    # Make predictions on the test partition
    # predictions = predict_model(model, data=partition_test)
    # Save the model and predictions for future use
    # save_model(model, f"model_{i}")
    # predictions.to_csv(f"predictions_{i}.csv", index=False)


# stop recording time and CPU utilization
# end_time = time.time()
# end_cpu_utilization = psutil.cpu_percent()
#
# # calculate training time and CPU utilization
# training_time = end_time - start_time
# cpu_utilization = end_cpu_utilization - start_cpu_utilization
#
# print("Training Time: ", training_time)
# print("Resource Utilization: ", cpu_utilization)

