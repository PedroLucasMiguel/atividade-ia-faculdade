from sklearn import svm

from ClassifierHandler import ClassifierHandler

MODEL_NAME = 'support_vector_machine'


def main():
    svm_classifier = svm.SVC(random_state=0)
    handler = ClassifierHandler(svm_classifier)
    handler.save_results(MODEL_NAME)
    handler.save_model(MODEL_NAME)
    print('Program Finished Successfully')


if __name__ == '__main__':
    main()
