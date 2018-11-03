Datos extraidos de la competición de Kaggle 'Digit Recognizer' (https://www.kaggle.com/c/digit-recognizer)

Han sido comprimidos utilizando gzip y para leerlos proceder como sigue (ejemplo para datos de training):

train_data_compressed = pd.read_csv("pytorchlib/pytorch_examples/data/MNIST/train.csv.gz", compression='gzip', index_col=0)
train_features_compressed = train_data_compressed.iloc[:, 1:].values.reshape(-1, 28, 28, 1)