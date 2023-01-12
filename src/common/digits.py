from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


def main():
    print('Hello, world!')
    digits = load_digits()
    print(digits.data.shape)
    plt.gray()
    plt.matshow(digits.images[0])
    plt.show()


if __name__ == '__main__':
    main()
