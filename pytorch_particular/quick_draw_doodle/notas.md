
9-11-2018
Probando entrenamiento con datos de menor a mayor a proporcionado peores resultados:
    #data_train_per_epoch_original = 145000
    data_amounts = ["1k", "10k", "50k"]
    epochs_per_amounts = [50, 20, 12]
    data_train_per_amounts = [40000, 75000, 100000]
    start_lr_amounts = [0.1, 0.085, 0.065]




---- MODELOS ----
MobileNetv2Standard con iamgenes de 64x64 -> last_pool_size = 8 y flatsize = 1280