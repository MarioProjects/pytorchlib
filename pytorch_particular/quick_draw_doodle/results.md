
---- DONE ----

|  Model Type |       Configuration      | Images Type |     Data Augmentation    |     Data Usage     | Accuracy@1 | Accuracy@3 | Leaderboard |
|:-----------:|:------------------------:|:-----------:|:------------------------:|:------------------:|:----------:|:----------:|:-----------:|

|   ResNet*   |   Bottle101_ExtraSmall   |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    76.30   |    90.39   |    -----    |
|   ResNet*   |       Basic34_Small      |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    77.22   |    91.34   |    0.887    | >>> steps with DA
|   ResNet*   |       Basic34_Small      |    64x64    |            No            | 50k - 145000 epoch |    78.99   |    91.68   |    0.892    | >>> steps lr 0.35
|   ResNet*   |       Basic34_Small      |    64x64    |            No            |  1k -> 10k -> 50k  |    74.80   |    89.80   |    -----    | >>> data amounts
|   ResNet*   |       Basic34_Small      |    64x64    |           HFlip          | 50k - 145000 epoch |    77.60   |    90.65   |    -----    | >>> scheduler
|   ResNet*   |       Basic34_Small      |    64x64    |           HFlip          | 50k - 145000 epoch |    76.68   |    90.85   |    -----    | >>> steps lr 0.1
|   ResNet*   |       Basic34_Small      |    64x64    |            No            | 50k - 200000 epoch |    79.37   |    92.00   |    0.894    | >>> RETRAIN steps 0.15...
---------------------------------------> (Basic34_Small) Al aplicar Test Time Augmentations (4 variantes + Original) paso de 0.894 a  0.888    | >>> TTA (Empeora)
|   ResNet    |   Basic34_Small_Color    |    64x64    |            No            | All - 200000 epoch |    78.11   |    91.80   |    0.891    | >>> Implicit retrain steps
|   ResNet    |     Basic34_Standard     |    64x64    |            No            | All - 200000 epoch |    78.71   |    91.85   |    0.888    | >>> Implicit retrain steps
|   ResNet    |        IMAGENET34        |   224x224   |            No            | All - 200000 epoch |    78.64   |    92.08   |             | >>> Imagenet weights
|   ResNet*   |     Basic18_Standard     |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    76.45   |    90.85   |    0.894    |
|   ResNet    |     Basic18_Standard     |    64x64    |            No            | All - 200000 epoch |    79.02   |    92.17   |    0.898    | >>> Implicit Retrain steps
---------------------------------------> (Basic18_Standard) Al aplicar Test Time Augmentations (3 HFlips + Original) paso de 0.894 a  0.897    | >>> TTA (Empeora)
|   ResNet    |     Basic18_Standard     |    64x64    |            No            | All - 200000 epoch |    79.11   |    92.17   |    0.895    | >>> I_Retrain TRAZOS3
|   ResNet    |     Basic18_Standard     |    64x64    |            No            | All - 200000 epoch |    79.26   |    92.34   |    0.893    | >>> I_Retrain TRAZOS3y2
----------------------> (Basic18_Standard) Al utilizar un batch mayor (512) el accuracy se ve reducido |    78.57   |    91.57   |    0.895    |
|   SENet*    |      PreAct34_Small      |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    77.60   |    90.65   |             |
|   SENet     |    PreAct50_Standard     |    64x64    |            No            | All - 200000 epoch |    79.57   |    92.34   |    ?????    | >>> Implicit retrain 0.1
|   SENet     |  PreAct50_Standard_Color |    64x64    |            No            | All - 200000 epoch |    80.37   |    93.03   |    0.914    | >>> Implicit retrain 0.1
|   SENet     |     PreAct50_Standard    |    64x64    |            No            | All - 200000 epoch |    80.26   |    93.03   |    0.909    | >>> I_Retrain TRAZOS3
------------------------------------------> Probando PreAct50_Small_Color con modelo mejor Acc1 y modelo mejor ACC3 obtienen mismo LB 0.914
|  SEResNext  |     Bottle50_Standard    |    64x64    |            No            | All - 200000 epoch |    76.91   |    90.97   |    0.885    | >>> Implicit retrain 0.1
|  SEResNext  |     Bottle50_Standard    |    64x64    |            No            | All - 200000 epoch |    77.74   |    91.29   |    0.882    | >>> I_Retrain TRAZOS3
|  SEResNext  |    Bottle101_Standard    |    64x64    |            No            | All - 200000 epoch |    76.66   |    90.43   |    -----    | >>> Implicit retrain 0.1
| MobileNetv2*|         Standard         |    64x64    |           HFlip          | 50k - 145000 epoch |    76.65   |    90.57   |             | >>> scheduler
| MobileNetv2*|         Standard         |    64x64    |            No            | 50k - 200000 epoch |    77.34   |    91.74   |    0.883    | >>> steps lr 0.35, 0.1 ...
|    VGG*     |      QDrawLargeVGG       |   224x224   |            No            | 50k - 200000 epoch |    71.65   |    86.57   |             |


-* Pytorchlib old version

---- RUNNING ----

|   Model Type  |     Configuration    | Images Type |     Data Augmentation    |     Data Usage     |    GPU    |
|:-------------:|:--------------------:|:-----------:|:------------------------:|:------------------:|:---------:|
|     SeNet     |  PreAct50_Standard   |    64x64    |            No            | All - 200000 epoch |    P12    | >>> Implicit retrain 0.1 Trazos3
se-resnext-50 (lw 3) -> gpu6
->> GPU10 Mixup res18


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


   -----> BASELINE PERFORMANCE:

      LB: 0.944

         se-resnext-50 (256x256 input, 200 samples per batch), single model, no TTA , no ensmble
         all train samples (simplified only, recognized and non-recognized): random 80 images per class are selected as validation, the rest are training samples.
         validation loss : ce_loss, top1, top3, (map@3)
         0.607 0.842 0.946 (0.890)*
         train loss : ce_loss, top1, top3, (map@3)
         0.590 0.844 0.948 (0.892)

      LB 0.928

         customized LSTM
         (Max seq length = 600, 200 samples per batch), single model, no TTA , no ensmble
         same train samples as above
         validation loss : ce_loss, top1, top3, (map@3)
         0.637 0.834 0.943 (0.885)*
         train loss : ce_loss, top1, top3, (map@3)
         0.595 0.841 0.949 (0.892)

      LB 0.945

         stacking via a fuse classifier on features from cnn and lstm. , no TTA , no ensmble
         same train samples as above. batch size increase to 512
         validation loss : ce_loss, top1, top3, (map@3)
         0.588 0.850 0.947 (0.895)*
         train loss : ce_loss, top1, top3, (map@3)
         0.549 0.850 0.949 (0.896)

