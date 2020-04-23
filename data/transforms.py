import numpy as np



class GaussianHeatMap(object):
    def __init__(self, size, sigma2=10, threshold=128):
        """
        :param size: array-like, ndim = 2 = (h, w)
        :param sigma2: variance, float
        :param threshold: float
        """
        self.size = size
        self.sigma2 = sigma2
        self.threshold = threshold

    @property
    def width(self):
        return self.size[1]
    @property
    def height(self):
        return self.size[0]

    def __call__(self, target):
        """
        :param target: ndarray, shape = (ball number, 2=(x, y))
        :return: heatmap ndarray, shape = (ball number, height, width)
        """
        ball_num = target.shape[0]

        x_step, y_step = np.arange(self.width), np.arange(self.height)
        x, y = np.meshgrid(x_step, y_step) # shape = (h, w)
        x, y = np.broadcast_to(x, (ball_num, self.height, self.width)), np.broadcast_to(y, (ball_num, self.height, self.width))

        balls = target.copy()
        balls[:, 0] *= self.width
        balls[:, 1] *= self.height
        balls = np.broadcast_to(balls, (ball_num, self.height, self.width, 2))

        exponent = -((np.power(x - balls[:, :, :, 0], 2) + np.power(y - balls[:, :, :, 1], 2))/(2*self.sigma2))
        g = np.exp(exponent) * 255

        g[g < self.threshold] = 0

        # sum and clamp
        g = np.sum(g, axis=0)
        g = np.clip(g, 0, 255)

        return g