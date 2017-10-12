import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


if __name__ == '__main__':
    columns = ['Stop distance', 'Speed']
    data = np.loadtxt('carstopping.txt', skiprows=1)
    speed = data[:, 1].reshape(63, 1)
    distance = data[:, 0].reshape(63, 1)

    regr = LinearRegression()
    regr.fit(speed, distance)
    pred_dist = regr.predict(speed)

    # for visualising fitted line
    fit_line_x = [[0], max(speed) + 10]  # speed range
    fit_line_y = regr.predict(fit_line_x)

    plt.figure(figsize=(12, 6))
    plt.title('$r^2$ = {:.2f}'.format(r2_score(distance, pred_dist)))

    plt.scatter(speed, distance)
    plt.plot(fit_line_x, fit_line_y, color='red')

    plt.xlabel(columns[1] + ', mph')
    plt.ylabel(columns[0] + ', feet')

    plt.savefig('fit_line.png')
    plt.show()
