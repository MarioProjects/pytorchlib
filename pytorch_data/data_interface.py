from .load_data import *

def database_selector(db, gray, seed=0, fold_test=0, batch_size=128, norm="None", show_info=True):
    train_samples, val_samples = False, False
    if db=="LFW":
        train_samples, train_loader, val_samples, val_loader = lfw_gender(gray, seed=seed, fold_test=fold_test, batch_size=batch_size, norm=norm)
    elif db=="LFW_Test":
        val_samples, val_loader = lfw_gender_test(gray, seed=seed, fold_test=fold_test, batch_size=batch_size, norm=norm)
    elif db =="GROUPS":
        train_samples, train_loader, val_samples, val_loader = groups_gender(gray, seed=seed, fold_test=fold_test, batch_size=batch_size, norm=norm)
    else: assert False, "Uknown database '" + str(db) + "'"

    if show_info:
        print("Database selected: {}, with gray flag {}, fold test {} and normalization {}".format(db, bool(gray), fold_test, norm))
        if train_samples: print('==>>> total trainning batch number: {}, with {} samples'.format(len(train_loader), train_samples))
        if val_samples: print('==>>> total validation batch number: {}, with {} samples'.format(len(val_loader), val_samples))
    
    if train_samples: return train_samples, train_loader, val_samples, val_loader
    else: return val_samples, val_loader