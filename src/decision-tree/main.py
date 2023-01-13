from sklearn import tree

ClassifierType = tree.DecisionTreeClassifier | None  # TODO: Add SVM


class ClassifierHandler:
    def __init__(self, classifier: ClassifierType):
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        self._classifier = classifier

        # Prepare train/test data
        digits = load_digits()
        num_samples = len(digits.images)
        # Flatten it
        data = digits.images.reshape((num_samples, -1))

        x_train, x_test, y_train, y_test = train_test_split(
            data, digits.target, shuffle=False
        )

        self._classifier.fit(x_train, y_train)

        self._true = y_test
        self._pred = self._classifier.predict(x_test)

    def make_classification_report(self) -> str:
        from sklearn.metrics import classification_report
        return classification_report(self._true, self._pred)

    def make_confusion_matrix(self) -> str:
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(self._true, self._pred)

    def calculate_loss(self) -> float:
        from sklearn.metrics import zero_one_loss
        return zero_one_loss(self._true, self._pred, normalize=True)


def main():
    tree_classifier = tree.DecisionTreeClassifier(random_state=0)
    handler = ClassifierHandler(tree_classifier)
    print(handler.make_classification_report())
    print(handler.make_confusion_matrix())
    print(f'Loss: {handler.calculate_loss()}')


if __name__ == '__main__':
    main()
