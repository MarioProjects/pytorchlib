from .load_data import *

def database_selector(db, gray=0, seed=0, fold_test=0, batch_size=128, norm=None, show_info=True, data_segment="0_10000", data_type="", transforms=[], NAME_TO_CLASS=""):
    train_samples, val_samples = False, False
    if db=="LFW":
        train_samples, train_loader, val_samples, val_loader = lfw_gender(gray, seed=seed, fold_test=fold_test, batch_size=batch_size, norm=norm)
    elif db=="LFW_Test":
        val_samples, val_loader = lfw_gender_test(gray, seed=seed, fold_test=fold_test, batch_size=batch_size, norm=norm)
    elif db =="GROUPS":
        train_samples, train_loader, val_samples, val_loader = groups_gender(gray, seed=seed, fold_test=fold_test, batch_size=batch_size, norm=norm)
    elif db =="QuickDraw":
        train_samples, train_loader = quick_draw_doodle(seed=seed, train_segment=data_segment, batch_size=batch_size, norm=norm, data_type=data_type, transforms=transforms, NAME_TO_CLASS=NAME_TO_CLASS)
    else: assert False, "Uknown database '" + str(db) + "'"

    if show_info:
        try:
            print("Database selected: {}, with gray flag {}, fold test {} and normalization {}".format(db, bool(gray), fold_test, norm))
            if train_samples:
                print('==>>> total trainning batch number: {}, with {} samples'.format(len(train_loader), train_samples))
            if val_samples:
                print('==>>> total validation batch number: {}, with {} samples'.format(len(val_loader), val_samples))
        except: pass 

    if train_samples and val_samples: return train_samples, train_loader, val_samples, val_loader
    if train_samples and not val_samples: return train_samples, train_loader
    else: return val_samples, val_loader