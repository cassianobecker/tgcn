class Layer:

    def __init__(self, val):
        self.val = val

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        print('The result is {}'.format(self.val + x))


if __name__ == '__main__':

    layer = Layer(5)
    layer(1)
