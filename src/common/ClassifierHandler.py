from sklearn import tree

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss
from joblib import dump

ClassifierType = tree.DecisionTreeClassifier | None  # TODO: Add SVM


class ClassifierHandler:
    def __init__(self, classifier: ClassifierType):
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
        return classification_report(self._true, self._pred)

    def make_confusion_matrix(self) -> str:
        return confusion_matrix(self._true, self._pred)

    def calculate_loss(self) -> float:
        return zero_one_loss(self._true, self._pred, normalize=True)

    def save_results(self, file_name: str):
        with open(f'../../generated/results/{file_name}.txt', 'w', encoding='UTF-8') as results_file:
            results_file.write(f"""
    Report:
    
{self.make_classification_report()}

    Confusion Matrix:
    
{self.make_confusion_matrix()}
    
    Zero-One Loss: {self.calculate_loss()}
    """)

    def save_model(self, file_name: str):
        path = f'../../generated/models/{file_name}.joblib'
        dump(self._classifier, path)
