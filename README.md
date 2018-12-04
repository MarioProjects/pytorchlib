# **Pytorchlib** - A Pytorch library with some helper functions

### Requisitos

Para la utilización de esta 'librería' es necesaria la instalación de algunas librerías de Python. Se recomienda la utilización de entornos virtuales:

```sh
cd ~
virtualenv -p python3.5 venv
source venv/bin/activate

python3 -m pip install numpy pandas scipy matplotlib sklearn scikit-learn scikit-image slackclient torch torchvision
python3 -m pip install opencv-python albumentations

# Había un problema con ipython y prompt-toolkit (quizás ya resuelto)
python3 -m pip uninstall ipython
python3 -m pip install ipython==6.5.0
python3 -m pip uninstall prompt-toolkit
python3 -m pip install prompt-toolkit==1.0.15

# ESTO AL BASH
echo ""  >> ~/.bashrc
echo ""  >> ~/.bashrc
echo "# MIS COSAS"  >> ~/.bashrc
echo "alias venv='source /home/maparla/venv/bin/activate'" >> ~/.bashrc
echo "venv"  >> ~/.bashrc
```

Además, si deseamos utilizar la funcionalidad de Slack ([para hacer el logging más fácil](https://github.com/MarioProjects/Python-Slack-Logging)) deberemos añadir al sistema el [token](https://github.com/MarioProjects/Python-Slack-Logging) de nuestro espacio de trabajo al sistema:

```sh
echo "export SLACK_TOKEN='my-slack-token'" >> ~/.bashrc
```

## Pytorch Library
Es el corazón de la librería. Contiene ayuda para la creación de las distintas redes, para el entrenamiento de las mismas así como funciones de ayuda más genéricas.


##### utils_general.py

   - ***slack_message***: Nos permite mandar un mensaje al canal de nuestra cuenta Slack que deseemos. Necesario añadir nuestro propio token personal en la variable con tal nombre. Como conseguir tu token [aqui](https://github.com/MarioProjects/Python-Slack-Logging).
   - ***time_to_human***: Dado un momento de inicio y final nos devuelve un string con las horas y minutos transcurridos entre ambos momentos.
   - ***images_to_vectors***: Transforma las imágenes que le pasamos (en formato Tensor de torch) a vectores.
   - ***vectors_to_images***: Transforma vectores (en formato Tensor de torch) a imágenes con las dimensiones especificadas.
   - ***normal_noise***: Con esta función creamos tantas muestras como deseemos formadas por ruido gausiano con una distribución normal con media 0 y varianza 1. Dimensionalidad también a indicar.

##### utils_training.py

  - ***to_categorical***: Convierte un vector con las clases a una matriz binaria de clases (codificación one-hot).
  - ***get_optimizer***: Crea un optimizador que deseemos. Ayuda para cambiar el learning rate durante el entrenamiento.
  - ***get_current_lr***: Dado un optimizador nos devuelve su learning rate actual.
  - ***anneal_lr_lineal***: Calcula siguiendo una función lineal el learning rate que debemos establecer al hacer learning rate annealing y si es el momento o no para cambiar el optimizador.
  - ***defrost_model_params***: 'Descongela' los parámetros de un modelo dado.
  - ***simple_target_creator***: Crea un vector lleno con el valor deseado. Util para crear vectores de targets.
  - ***train_simple_model***: Dados un modelo, su optimizador y una serie de datos de entrenamiento, realiza el forward a los datos y optimizados el modelo en función a función de coste especificada.
  - ***evaluate_accuracy_models***: Calcula la tasa de acierto de los modelos sobre una serie de datos, ya sean datos con sus respectivas etiquetas o dataloaders de Pytorch. Puede devolver el topk accuracy.
  - ***evaluate_accuracy_model_predictions***: Toma la salida de un modelo y los targets esperados y calcula la tasa de acierto correspondiente. Puede devolver el topk accuracy.
  - ***predictions_models_data***: Devuelve el forward obtenido por los modelos que le pasemos con los datos especificados.
  - ***topk_accuracy***: Calcula la tasa de acierto sobre las top k predicciones sobre los targets especificados.
  - ***train_discriminator***: Función para el entrenamiento genérico de un discriminador en el esquema de las 'Generative Adversarial Networks' (GAN).
  - ***train_generator***: Función para el entrenamiento genérico de un generador en el esquema de las 'Generative Adversarial Networks' (GAN).


##### utils_nets.py

  - ***get_activation***: Devuelve la función de activación que indiquemos.
  - ***apply_pool***: Devuelve una función pooling indicada.
  - ***apply_linear***: Define una secuencia lineal con un número de neuronas de entrada y salida especificadas, con la posibilidad de utilizar ruido gaussiano, Dropout, Batch Normalization...
  - ***apply_conv***: Define una secuencia convolucional con un número de mapas de entrada y salida deseado, con la posibilidad de utilizar ruido gaussiano, Dropout, Batch Normalization...
  - ***apply_DeConv***: Define una secuencia deconvolucional con un número de mapas de entrada y salida deseado, con la posibilidad de utilizar ruido gaussiano, Dropout, Batch Normalization...
  - ***apply_DePool***: Para realizar upsampling con el kernel especificado.
  - ***topk_classes***: Dada una entrada con diferentes probabilidades, devuelve las k posiciones de aquellas de mayor probabilidad.
  - ***models_average***: Dadas una lista de salidas de probabilidades promedia dichas salidas siguiendo dos esquemas, votación y suma de probabilidades.

## Pytorch Models

##### basic_nets.py

  - ***MLPNet***: Módulo para la generación de Perceptrones Multicapa siguiendo una lista de configuración especificada.
  - ***ConvNet***: Módulo para la generación de redes convolucionales simples siguiendo una lista de configuración especificada.


## Pytorch Data
##### load_data.py
  - ***load_img***: Carga una imagen dada la ruta a ella en formato numpy.
  - ***FoldersDataset***: Dada la ruta hacia una carpeta donde almacenamos nuestra base de datos en imágenes con la estructura de data_path/clase1/imágenes*, data_path/clase2/imágenes*... Crea un Dataset de Pytorch con dichas imágenes.
  - ***normalize***: Dados dos Tensores (Pytorch) de datos, como podrían ser el conjunto de test y entrenamiento, los normaliza siguiendo alguna de las normalizaciones implementadas.
  - ***single_normalize***: Normaliza un Tensor (Pytorch) de datos, como podrían ser el conjunto de test o entrenamiento, lo normaliza siguiendo alguna de las normalizaciones implementadas.
  - ***apply_img_albumentation***: Nos sirve para aplicar una [albumentation](https://github.com/albu/albumentations) a una imagen dada en formato numpy.

## Pytorch Examples

  - ***ce_simple_scheduler_MNIST.py***: Ejemplo de entrenamiento de un modelo MLP utilizando cros-entropía sobre el problema de MNIST a partir de los datos proporcionados por Pytorch. Se utiliza un scheduler de Pytorch para la reducción periódica del learning rate cuando la tasa de acierto se estanca.
  - ***ce_simple_steps_CIFAR10.py***: Entrenamiento de un modelo de VGG utilizando la cros-entropía para el problema de CIFAR-10 a partir de los datos proporcionados por Pytorch. Para la reducción del learning rate se emplea un esquema en el que lo reducimos cada cierto número de epochs.



License & Credits
----
Gracias a [Kuangliu](https://github.com/kuangliu) por la implementación de los modelos y a [Juan Maroñas](https://github.com/jmaronas) por su ayuda en la creación de diversas funciones.

MIT - **Free Software, Hell Yeah!**


