import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_x = train_df['x'].values
train_r = train_df['r'].values
test_x = test_df['x'].values
test_r = test_df['r'].values

train_sse = []
test_sse = []

degrees = range(7)

for degree in degrees:
    coeffs = np.polyfit(train_x, train_r, degree)
    p = np.poly1d(coeffs)

    train_predictions = p(train_x)
    train_sse.append(np.sum((train_predictions - train_r) ** 2))
    test_predictions = p(test_x)
    test_sse.append(np.sum((test_predictions - test_r) ** 2))

    smooth_train_x = np.linspace(train_x.min(), train_x.max(), num=100)
    smooth_test_x = np.linspace(test_x.min(), test_x.max(), num=100)

    train_y_vals = p(smooth_train_x)
    test_y_vals = p(smooth_test_x)

    plt.scatter(train_x, train_r, label='Training data')
    plt.plot(smooth_train_x, train_y_vals, 'r',
             label='Degree = {}'.format(degree))
    plt.legend()
    plt.show()

    plt.scatter(test_x, test_r, label='Test data')
    plt.plot(smooth_test_x, test_y_vals, 'm',
             label='Degree = {}'.format(degree))
    plt.legend()
    plt.show()

plt.figure()
plt.plot(degrees, train_sse, 'o-', label='Training SSE')
plt.plot(degrees, test_sse, 'o-', label='Test SSE')
plt.xlabel('Polynomial degree')
plt.ylabel('SSE')
plt.title('SSE on Training and Test Sets')
plt.legend()
plt.show()
