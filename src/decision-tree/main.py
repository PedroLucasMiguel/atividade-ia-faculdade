from sklearn import tree

from common.ClassifierHandler import ClassifierHandler

MODEL_NAME = 'decision_tree'


def main():
    tree_classifier = tree.DecisionTreeClassifier(random_state=0)
    handler = ClassifierHandler(tree_classifier)
    handler.save_results(MODEL_NAME)
    handler.save_model(MODEL_NAME)
    print('Program Finished Successfully')


if __name__ == '__main__':
    main()
