
---- DONE ----

|  Model Type |     Configuration    | Images Type |     Data Augmentation    |     Data Usage     | Accuracy@1 | Accuracy@3 | Leaderboard |
|:-----------:|:--------------------:|:-----------:|:------------------------:|:------------------:|:----------:|:----------:|:-----------:|

|   ResNet*   | Bottle101_ExtraSmall |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    76.30   |    90.39   |    -----    |
|   ResNet*   |     Basic34_Small    |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    77.22   |    91.34   |    0.887    | >>> steps with DA
|   ResNet*   |     Basic34_Small    |    64x64    |            No            | 50k - 145000 epoch |    78.99   |    91.68   |    0.892    | >>> steps lr 0.35
|   ResNet*   |     Basic34_Small    |    64x64    |            No            |  1k -> 10k -> 50k  |    74.80   |    89.80   |    -----    | >>> data amounts
|   ResNet*   |     Basic34_Small    |    64x64    |           HFlip          | 50k - 145000 epoch |    77.60   |    90.65   |    -----    | >>> scheduler
|   ResNet*   |     Basic34_Small    |    64x64    |           HFlip          | 50k - 145000 epoch |    76.68   |    90.85   |    -----    | >>> steps lr 0.1
|   ResNet*   |     Basic34_Small    |    64x64    |            No            | 50k - 200000 epoch |    79.37   |    92.00   |    0.894    | >>> RETRAIN steps lr 0.15...
-----------------------------------> (Basic34_Small) Al aplicar Test Time Augmentations (4 variantes + Original) paso de 0.894 a  0.888    | >>> TTA (Empeora)
|   ResNet    | Basic34_Small_Color  |    64x64    |            No            | All - 200000 epoch |    78.11   |    91.80   |    0.891    | >>> Implicit retrain steps
|   ResNet    |    Basic34_Standard  |    64x64    |            No            | All - 200000 epoch |    78.71   |    91.85   |    0.888    | >>> Implicit retrain steps
|   ResNet    |      IMAGENET34      |   224x224   |            No            | All - 200000 epoch |    78.64   |    92.08   |             | >>> Retrain Imagenet weights
|   ResNet*   |   Basic18_Standard   |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    76.45   |    90.85   |    0.894    |
|   ResNet    |   Basic18_Standard   |    64x64    |            No            | All - 200000 epoch |    79.02   |    92.17   |    0.898    | >>> Implicit Retrain steps
-----------------------------------> (Basic18_Standard) Al aplicar Test Time Augmentations (3 HFlips + Original) paso de 0.894 a  0.897    | >>> TTA (Empeora)
------------------> (Basic18_Standard) Al utilizar un batch mayor (512) el accuracy se ve reducido |    78.57   |    91.57   |    0.895    |
|   SENet*    |    PreAct34_Small    |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    77.60   |    90.65   |             |
| MobileNetv2*|        Standard      |    64x64    |           HFlip          | 50k - 145000 epoch |    76.65   |    90.57   |             | >>> scheduler
| MobileNetv2*|        Standard      |    64x64    |            No            | 50k - 200000 epoch |    77.34   |    91.74   |    0.883    | >>> steps lr 0.35, 0.1 ...
|   VGG*      |     QDrawLargeVGG    |   224x224   |            No            | 50k - 200000 epoch |    71.65   |    86.57   |             |


-* Pytorchlib old version

---- RUNNING ----

|   Model Type  |     Configuration    | Images Type |     Data Augmentation    |     Data Usage     |    GPU    |
|:-------------:|:--------------------:|:-----------:|:------------------------:|:------------------:|:---------:|
|               |                      |             |                          | All - 200000 epoch |           | >>> 


---- PENDING ----


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


  ->"Augmentation is usually better for small training dataset …"
        Not true.
        For this challenge, you should look at the train and test images and think of how to make the train more similar to the test?
        Especially, check those mis-classified validation images. why are them different from the training set?
        e.g.
        incomplete drawing : hence drop stroke can be a good augmentation
        inclusion of text and arrow : hence add text or arrow can be a good augmentation 