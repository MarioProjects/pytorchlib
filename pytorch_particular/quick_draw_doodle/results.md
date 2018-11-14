
---- DONE ----

|  Model Type |     Configuration    | Images Type |     Data Augmentation    |     Data Usage     | Accuracy@1 | Accuracy@3 | Leaderboard |
|:-----------:|:--------------------:|:-----------:|:------------------------:|:------------------:|:----------:|:----------:|:-----------:|
|   ResNet    | Bottle101_ExtraSmall |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    76.30   |    90.39   |    -----    |
|   ResNet    |     Basic34_Small    |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    77.22   |    91.34   |    0.887    | >>> steps with DA
|   ResNet    |     Basic34_Small    |    64x64    |            No            | 50k - 145000 epoch |    78.99   |    91.68   |    0.892    | >>> steps lr 0.35
|   ResNet    |     Basic34_Small    |    64x64    |            No            |  1k -> 10k -> 50k  |    74.80   |    89.80   |    -----    | >>> data amounts
|   ResNet    |     Basic34_Small    |    64x64    |           HFlip          | 50k - 145000 epoch |    77.60   |    90.65   |    -----    | >>> scheduler
|   ResNet    |     Basic34_Small    |    64x64    |           HFlip          | 50k - 145000 epoch |    76.68   |    90.85   |    -----    | >>> steps lr 0.1
|   ResNet    |     Basic34_Small    |    64x64    |            No            | 50k - 200000 epoch |    79.37   |    92.00   |    0.894    | >>> RETRAIN steps lr 0.15...
|   ResNet    |   Basic18_Standard   |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    76.45   |    90.85   |    0.894    |
|   SENet     |    PreAct34_Small    |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    77.60   |    90.65   |             |
| MobileNetv2 |        Standard      |    64x64    |           HFlip          | 50k - 145000 epoch |    76.65   |    90.57   |             | >>> scheduler
| MobileNetv2 |        Standard      |    64x64    |            No            | 50k - 200000 epoch |    77.34   |    91.74   |    0.883    | >>> steps lr 0.35, 0.1 ...
|   VGG       |     QDrawLargeVGG    |   224x224   |            No            | 50k - 200000 epoch |    71.65   |    86.57   |             |


---- RUNNING ----

|   Model Type  |     Configuration    | Images Type |     Data Augmentation    |     Data Usage     |    GPU    |
|:-------------:|:--------------------:|:-----------:|:------------------------:|:------------------:|:---------:|
|               |                      |             |                          |                    |           |


---- PENDING ----

  --> Mobilenet Standard sin DA, steps, larger LR y larger data per epoch REENTRENADO del que ya tengo
  --> REENTRENAR ResNet Basic18 Standard?
  --> Probar a incrementar el batch size haciendo que el coste se vaya aumgentando y posteriormente .backward()
  --> Utilizar imagenes a color!
  --> Reentrenar modelo de Imagenet

---- IDEAS Y CONCLUSIONES ----

  x> Debido al gran numero de muestras la gente propone no utilizar Data Augmentation
  x> Parace que los Learning rates altos funcionan mejor: 0.35 -> 0.1 -> ...
  
  -> Comentan que como los datos son ruidosos utilizar batch size altos (512 o más)...
     como nuestras GPUs explotan normalmente con este abtch size, podemos ir sumando costes
     hasta haber sumado tantas como batch deaseado y entonces hacer el backward
  -> SENet sin Data Augmentation
  -> Ensembling :
       ---> Probar a promediar resultados y crear baseline
       ---> Hacer un MLP simple que tome las salidas de varios clasificadores y entrenarlo

  ->Could you tell more about "you have 3 channel rgb, make good use of it! try to encode more information"? you mean use the bigger filters?
     ->Try using stroke velocity in 2nd channel and perhaps stroke acceleration in the 3rd channel
     ->There's no temporal info about the different points on a stroke though
     ->There are time stamps for each point in raw data
 