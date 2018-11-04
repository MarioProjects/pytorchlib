Datos extraidos de la competición de Kaggle 'Digit Recognizer' (https://www.kaggle.com/c/digit-recognizer)

Para leerlos proceder como sigue (ejemplo para datos de training):

train_data = pd.read_csv("train.csv")
train_features = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)