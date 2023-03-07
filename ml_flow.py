import mlflow

EXPERIMENT_NAME = 'Experiments'

RUN_NAME = 'Demo_Project'

class Mlflow:
    def __init__(self, model, test_acc, train_acc):
        self.model = model
        self.test_acc = test_acc
        self.train_acc = train_acc

    def create_experiment(self):
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run():

            mlflow.log_metric('accuracy_test',self.test_acc)
            mlflow.log_metric('accuracy_train',self.train_acc)

            mlflow.sklearn.log_model(self.model, 'model')

            mlflow.set_tag('tag1','Demo-Project')
            mlflow.set_tag('model','logistic regression')
