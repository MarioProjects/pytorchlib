
---- DONE ----

| Model Type |     Configuration    | Images Type |     Data Augmentation    |     Data Usage     | Accuracy@1 | Accuracy@3 | Leaderboard |
|:----------:|:--------------------:|:-----------:|:------------------------:|:------------------:|:----------:|:----------:|:-----------:|
|   ResNet   | Bottle101_ExtraSmall |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    76.30   |    90.39   |    -----    |
|   ResNet   |     Basic34_Small    |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    77.22   |    91.34   |    0.887    |
|   ResNet   |     Basic34_Small    |    64x64    |            No            | 50k - 145000 epoch |    78.99   |    91.68   |    0.892    |
|   ResNet   |     Basic34_Small    |    64x64    |            No            |  1k -> 10k -> 50k  |    74.80   |    89.80   |    -----    |
|   ResNet   |   Basic18_Standard   |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    76.45   |    90.85   |    0.894    |
|   SENet    |    PreAct34_Small    |    64x64    | HFlip & ShiftScaleRotate | 50k - 145000 epoch |    77.60   |    90.65   |             |



---- RUNNING ----

|   Model Type  |     Configuration    | Images Type |     Data Augmentation    |     Data Usage     |    GPU    |
|:-------------:|:--------------------:|:-----------:|:------------------------:|:------------------:|:---------:|
|     ResNet    |     Basic34_Small    |    64x64    |          HFlip           | 50k - 145000 epoch |    P12    |
|  MobileNetv2  |       Standard       |    64x64    |          HFlip           | 50k - 145000 epoch |    P12    |
|      VGG      |     QDrawLargeVGG    |   224x224   |          None            | 50k - 200000 epoch |    P11    |


---- PENDING ----

|      VGG      |     QDrawSmallVGG    |    32x32    |          HFlip           | 50k - 350000 epoch |    P11    |
|      VGG      |     QDrawSmallVGG    |    64x64    |          HFlip           | 50k - 350000 epoch |    P11    |
|      VGG      |     QDrawMediumVGG   |    32x32    |          HFlip           | 50k - 350000 epoch |    P11    |