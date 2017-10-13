import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def createFittedLinePoints(rng, model):
    '''
        returns points of line for plt.plot [[x1, x2], [y1, y2]]
    '''
    xs = np.array(rng).reshape(2, 1)
    ys = model.predict(xs)
    return [xs, ys]


if __name__ == '__main__':
    columns = ['McCoo rush distance', 'Game scores']
    data = np.loadtxt('mccoo.txt', skiprows=1)

    distance = data[:, 0].reshape(data.shape[0], 1)
    score = data[:, 1].reshape(data.shape[0], 1)

    regr_all = LinearRegression()
    regr_all.fit(distance, score)
    pred_all_score = regr_all.predict(distance)

    regr_without_last = LinearRegression()
    regr_without_last.fit(distance[:-1], score[:-1])
    pred_without_last_score = regr_all.predict(distance[:-1])

    plt.figure(figsize=(12, 6))
    plt.title('All $r^2$ = {:.2f}, Without red $r^2$ = {:.2f}'
              .format(r2_score(score, pred_all_score),
                      r2_score(score[:-1], pred_without_last_score))
              )

    plt.scatter(distance, score)
    plt.plot(distance[-1], score[-1], 'ro')  # separated point

    line_x_range = [0, np.max(distance) + 10]
    plt.plot(*createFittedLinePoints(line_x_range, regr_all),
             color='green', label='All points')
    plt.plot(*createFittedLinePoints(line_x_range, regr_without_last),
             color='red', label='Without red point')

    plt.xlabel(columns[0] + ', yard')
    plt.ylabel(columns[1])
    plt.legend(loc='best')

    plt.savefig('fit_line.png')
    plt.show()
