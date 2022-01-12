import janestreet
import numpy as np
import pandas as pd
# import datatable as dt
import logging
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import time
from tqdm import tqdm

logging.getLogger('tensorflow').setLevel(logging.ERROR)

SUBMODEL_NUM = 6

loaded_submodel = []
for idx in range(SUBMODEL_NUM):
    path = "../input/marketmodelbv11/model_b/submodel_" + str(idx)
    loaded_submodel.append(tf.saved_model.load(path))
loaded_model = tf.saved_model.load("../input/marketmodelbv11/model_b/model")
# scaler = joblib.load('../input/marketmodelv3/model_b/std_scaler.bin')

env = janestreet.make_env()
iter_test = env.iter_test()

print("start inference")
start_time = time.time()
for (test_df, pred_df) in iter_test:
    test_np = test_df.to_numpy()
    if test_np[0, 0] > 0:
        x_tt = test_np[:, 1:131].astype(np.float32)
        np.nan_to_num(x_tt, copy=False)
        # x_tt = scaler.transform(x_tt)

        pred_sub_list = []
        for i in range(SUBMODEL_NUM):
            pred_sub = loaded_submodel[i](tf.constant(x_tt), training=False)
            pred_sub = tf.math.reduce_mean(pred_sub, axis=1)
            pred_sub = tf.expand_dims(pred_sub, axis=1)
            pred_sub_list.append(pred_sub)
        pred = loaded_model(pred_sub_list, training=False)
        # print(pred)
        pred_df.action = np.where(pred > 0.5, 1, 0).astype(int)
    else:
        pred_df.action = 0
    # print(pred_df.action)
    env.predict(pred_df)
print(f"took: {time.time() - start_time} seconds")