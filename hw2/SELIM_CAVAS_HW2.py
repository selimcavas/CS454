import pandas as pd
import math

# Read the CSV file
train_df = pd.read_csv('train_iris.csv')
test_df = pd.read_csv('test_iris.csv')

means_arr = []
for i in range(3):
    group_df = train_df[train_df['y'] == i]
    x1_mean = sum(group_df['x1']) / len(group_df)
    x2_mean = sum(group_df['x2']) / len(group_df)
    x3_mean = sum(group_df['x3']) / len(group_df)
    x4_mean = sum(group_df['x4']) / len(group_df)
    means_arr.append([x1_mean, x2_mean, x3_mean, x4_mean])

# Convert the train dataframe to a numpy array
train_arr = train_df[['x1', 'x2', 'x3', 'x4', 'y']].values.tolist()

# Print mean values for each group y
for i, row in enumerate(means_arr):
    x1 = round(row[0], 2)
    x2 = round(row[1], 2)
    x3 = round(row[2], 2)
    x4 = round(row[3], 2)
    print(
        f"Mean values for y = {i}: [{x1} {x2} {x3} {x4}]")

predictions = []
predictions_neighbour = []
for _, row in test_df.iterrows():
    x1_test = row['x1']
    x2_test = row['x2']
    x3_test = row['x3']
    x4_test = row['x4']
    min_distance = math.inf
    predicted_y = -1
    min_distance_neighbour = math.inf
    predicted_y_n = -1
    # Calculate distances between test points and each mean value, and make predictions based on closest mean
    for i, mean in enumerate(means_arr):
        x1_avg = mean[0]
        x2_avg = mean[1]
        x3_avg = mean[2]
        x4_avg = mean[3]
        distance = math.sqrt((x1_avg - x1_test)**2 + (x2_avg - x2_test)
                             ** 2 + (x3_avg - x3_test)**2 + (x4_avg - x4_test)**2)
        if distance < min_distance:
            min_distance = distance
            predicted_y = i
    # Calculate distances between test points and each train point, and make predictions based on closest train point
    for i, train in enumerate(train_arr):  # here
        x1_train = train[0]
        x2_train = train[1]
        x3_train = train[2]
        x4_train = train[3]
        distance = math.sqrt((x1_train - x1_test)**2 + (x2_train - x2_test)
                             ** 2 + (x3_train - x3_test)**2 + (x4_train - x4_test)**2)
        if distance < min_distance_neighbour:
            min_distance_neighbour = distance
            predicted_y_n = int(train[4])
    predictions.append(predicted_y)
    predictions_neighbour.append(predicted_y_n)

# Create confusion matrixes
confusion_matrix = [[0 for _ in range(3)] for _ in range(3)]
confusion_matrix_neighbour = [[0 for _ in range(3)] for _ in range(3)]
for index, row in test_df.iterrows():
    actual_y = int(row['y'])
    predicted_y = predictions[index]
    predicted_neighbour_y = predictions_neighbour[index]
    confusion_matrix[actual_y][predicted_y] += 1
    confusion_matrix_neighbour[actual_y][predicted_neighbour_y] += 1

print("\nConfusion matrix for Nearest-Mean classifier:")
print("   | 0 | 1 | 2 |")
print("---|---|---|---|")
for i in range(3):
    row = confusion_matrix[i]
    row_str = f"{i}  |"
    for item in row:
        row_str += f" {item} "
    print(row_str)

print("\nConfusion matrix: for Nearest-Neighbour classifier")
print("   | 0 | 1 | 2 |")
print("---|---|---|---|")
for i in range(3):
    row = confusion_matrix_neighbour[i]
    row_str = f"{i}  |"
    for item in row:
        row_str += f" {item} "
    print(row_str)

total_instances = len(test_df)
misclassified_instances = 0
misclassified_instances_n = 0

for i in range(3):
    for j in range(3):
        if i != j:
            misclassified_instances += confusion_matrix[i][j]
            misclassified_instances_n += confusion_matrix_neighbour[i][j]

misclassification_percentage = misclassified_instances / total_instances * 100
misclassification_percentage_n = misclassified_instances_n / total_instances * 100
print(
    f"\nOverall percentage of misclassified instances in Nearest-Mean classifier: {misclassification_percentage:.2f}%")
print(
    f"Overall percentage of misclassified instances in Nearest-Neighbour classifier: {misclassification_percentage_n:.2f}%")
