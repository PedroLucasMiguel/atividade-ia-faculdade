from sklearn.datasets import load_digits
from sklearn import tree
import matplotlib.pyplot as plt


def main():
    digits = load_digits()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(digits.data, digits.target)
    tree.plot_tree(clf)
    plt.show()


if __name__ == '__main__':
    main()
