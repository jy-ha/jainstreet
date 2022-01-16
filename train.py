from google.colab import drive
from google.colab import auth
drive.mount('/content/drive')
auth.authenticate_user()

!pip install datatable
!pip install tensorflow_addons
!pip install --upgrade 'scikit-learn==0.23.2'

import psutil
import os
import gc
pid = os.getpid()
current_process = psutil.Process(pid)

import time
import logging
import numpy as np
import pandas as pd
import datatable as dt
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm.notebook import tqdm

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras import Model

#_ = np.seterr(divide='ignore', invalid='ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def memory_state():
    current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
    print(f"Current memory KB   : {current_process_memory_usage_as_KB: 9f} KB")


# TPU setup

TPU = True
if TPU:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)


# Data Managing

class DataHandling():
    target = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
    batch_size = 2048
    shuffle_size = 3000000

    def __init__(self, max_train_rows, cv_frac=0.1):
        self.cross_valid_frac = cv_frac
        self.read_data(max_train_rows)
        memory_state()

        self.process_data()
        self.read_cluster()
        memory_state()

    def read_data(self, max_train_rows):
        # pd.read_csv(path, usecols=set(type_dict.keys()), nrows=num_rows, dtype=type_dict)
        # self.train_df = dt.fread('/content/drive/MyDrive/Colab Notebooks/kaggle_market/datafiles/train.csv', columns=set(self.train_data_types_dict.keys()), max_nrows=max_train_rows).to_pandas()
        self.train_df = dt.fread('/content/drive/MyDrive/Colab Notebooks/kaggle_market/datafiles/train.csv',
                                 max_nrows=max_train_rows).to_pandas()
        self.train_df = self.train_df.astype('float32')

    def read_cluster(self):
        cluster_path = [
            '/content/drive/MyDrive/Colab Notebooks/kaggle_market/feature_corr_clust_dn_update.txt',
            '/content/drive/MyDrive/Colab Notebooks/kaggle_market/feature_corr_clust_km_update.txt',
            '/content/drive/MyDrive/Colab Notebooks/kaggle_market/feature_corr_spr_clust_dn.txt',
            '/content/drive/MyDrive/Colab Notebooks/kaggle_market/feature_tag_lsi.txt',
            # '/content/drive/MyDrive/Colab Notebooks/kaggle_market/feature_tags_clust.txt',
            '/content/drive/MyDrive/Colab Notebooks/kaggle_market/feature_clust_db.txt'
        ]
        cluster_features = ['feature_list', 'feature_count']

        self.clusters = []
        self.clusters_cnt = []
        self.num_cluster = len(cluster_path)

        for path in cluster_path:
            df = pd.read_csv(path, usecols=set(cluster_features))
            self.clusters_cnt.append(df['feature_count'].to_list())
            self.clusters.append(df['feature_list'].str.split(', ').to_list())

        for idx_c, cluster in enumerate(self.clusters):
            for idx_fs, features in enumerate(cluster):
                for idx_f, feature in enumerate(features):
                    self.clusters[idx_c][idx_fs][idx_f] = self.features.index(feature)

    def process_data(self):
        # preprocess
        self.train_df.fillna(0, inplace=True)
        # self.train_df.fillna(self.train_df.mean(),inplace=True)
        # self.train_df['action'] = ((self.train_df['resp'].values) > 0).astype(int)

        self.features = [c for c in self.train_df.columns if "feature" in c]
        self.feature_num = len(self.features)

        # Standarize
        # scaler = joblib.load('/content/drive/MyDrive/Colab Notebooks/kaggle_market/blonix/std_scaler.bin')
        # self.train_df[self.features] = scaler.transform(self.train_df[self.features])

        # self.valid_df = self.train_df.groupby('date').sample(frac=self.cross_valid_frac)
        # self.train_df.drop(self.valid_df.index, inplace=True)
        train_count = self.train_df['resp'].agg('count')
        train_count = train_count * (1.0 - self.cross_valid_frac)
        self.valid_df = self.train_df[(self.train_df['ts_id'] > train_count)]
        self.train_df = self.train_df[(self.train_df['ts_id'] <= train_count)]
        # self.train_df = self.train_df[self.train_df['weight'] != 0]
        # self.valid_df = self.valid_df[self.valid_df['weight'] != 0]

    def get_dataset(self):
        train_x = np.array(self.train_df[self.features]).astype(np.float32)
        train_y = np.stack([(self.train_df[c] > 0).astype(int) for c in self.target]).T
        valid_x = np.array(self.valid_df[self.features]).astype(np.float32)
        valid_y = np.stack([(self.valid_df[c] > 0).astype(int) for c in self.target]).T

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
        train_ds = train_ds.shuffle(buffer_size=self.shuffle_size).batch(self.batch_size)
        valid_ds = valid_ds.shuffle(buffer_size=self.shuffle_size).batch(self.batch_size)
        train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_ds, valid_ds


# Layers

class ClusterDenseLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_nodes, output_nodes):
        super(ClusterDenseLayer, self).__init__()
        self.depth = 1
        self.dense_1 = []
        self.batch_normalization = []
        self.activation = []
        self.dropout = []
        self.batch_normalization.append(BatchNormalization())
        self.dropout.append(Dropout(0.2))
        for i in range(self.depth):
            self.dense_1.append(Dense(hidden_nodes))
            self.batch_normalization.append(BatchNormalization())
            self.activation.append(Activation(tf.keras.activations.swish))
            self.dropout.append(Dropout(0.2))
        self.dense_2 = Dense(output_nodes)
        self.activation_2 = Activation(tf.keras.activations.swish)

    def call(self, x):
        x = self.dropout[0](x)
        x = self.batch_normalization[0](x)
        for i in range(self.depth):
            x = self.dense_1[i](x)
            x = self.batch_normalization[i + 1](x)
            x = self.activation[i](x)
            x = self.dropout[i + 1](x)
        x = self.dense_2(x)
        out = self.activation_2(x)
        return out


class SubModel_Cluster(Model):
    def __init__(self, cluster, cluster_cnt):
        super(SubModel_Cluster, self).__init__()
        self.cluster = cluster
        self.cluster_cnt = cluster_cnt
        self.feature_total = sum(self.cluster_cnt)
        self.hidden_nodes_group = []
        self.output_num_group = []

        for cnt in self.cluster_cnt:
            hidden_nodes = (cnt / self.feature_total) * 520
            output_num = (cnt / self.feature_total) * 65
            if output_num < 1:
                output_num = 1
            self.hidden_nodes_group.append(hidden_nodes)
            self.output_num_group.append(output_num)

        self.cluster_dense_layers = []
        for idx_fs, features in enumerate(cluster):
            # self.cluster_cnt[idx_fs] # feature num
            new_c_d = ClusterDenseLayer(self.hidden_nodes_group[idx_fs], self.output_num_group[idx_fs])
            self.cluster_dense_layers.append(new_c_d)

        self.hidden_nodes_sub = [256, 256, 128]
        self.dense_1 = []
        self.batch_normalization = []
        self.activation = []
        self.dropout = []
        self.depth = len(self.hidden_nodes_sub)
        for i in range(self.depth):
            self.dense_1.append(Dense(
                self.hidden_nodes_sub[i]))  # activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)
            self.batch_normalization.append(BatchNormalization())
            self.activation.append(Activation(tf.keras.activations.swish))
            self.dropout.append(Dropout(0.2))
        self.dense_2 = Dense(5)
        self.activation_2 = Activation("sigmoid")

    def call(self, x, training=None):

        x_fs_list = []
        for idx_fs, features in enumerate(self.cluster):
            x_fs = tf.gather(x, features, axis=1)
            x_fs = self.cluster_dense_layers[idx_fs](x_fs, training=training)
            x_fs_list.append(x_fs)
        x = tf.concat(x_fs_list, 1)

        for i in range(self.depth):
            x = self.dense_1[i](x)
            x = self.batch_normalization[i](x)
            x = self.activation[i](x)
            x = self.dropout[i](x)
        x = self.dense_2(x)
        out = self.activation_2(x)
        return out


# Model

class MarketModel_A(Model):
    def __init__(self):
        super(MarketModel_A, self).__init__()

        self.hidden_nodes_sub = [150, 150, 150]
        self.dense_1 = []
        self.batch_normalization = []
        self.activation = []
        self.dropout = []
        self.depth = len(self.hidden_nodes_sub)

        self.dropout.append(Dropout(0.2))
        self.batch_normalization.append(BatchNormalization())
        for i in range(self.depth):
            self.dense_1.append(Dense(self.hidden_nodes_sub[i])) # activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)
            self.batch_normalization.append(BatchNormalization())
            self.activation.append(Activation(tf.keras.activations.swish))
            self.dropout.append(Dropout(0.2))
        self.dense_2 = Dense(5)
        self.activation_2 = Activation("sigmoid")

    def call(self, x, training=None):
        x = self.dropout[0](x)
        x = self.batch_normalization[0](x)
        for i in range(self.depth):
            x = self.dense_1[i](x)
            x = self.batch_normalization[i+1](x)
            x = self.activation[i](x)
            x = self.dropout[i+1](x)
        x = self.dense_2(x)
        out = self.activation_2(x)
        return out

class MarketModel_B(Model):
    def __init__(self, cluster_num):
        super(MarketModel_B, self).__init__()
        self.cluster_num = cluster_num

        #self.cluster_weights = self.add_weight(name='weight_linear', shape=(self.cluster_num, 1), initializer='uniform', trainable=True)
        self.cluster_weights = tf.Variable([[1.0] * self.cluster_num], trainable=True)
        self.softmax_out = tf.keras.layers.Softmax(axis=-1)
        #self.dropout = Dropout(0.2)
        #self.d_out = Dense(1, activation='linear') # kernel_regularizer=tf.keras.regularizers.l2(0.001)
        #self.d2 = Dense(2, activation='softmax')

    def call(self, x_list, training=None):
        out = tf.concat(x_list, 1)
        cluster_weights = self.softmax_out(self.cluster_weights)
        cluster_weights = tf.transpose(cluster_weights)
        out =  tf.matmul(out, cluster_weights)
        #out = tf.keras.layers.Lambda(lambda x: x / self.cluster_num)(out)
        return out


# Train

LEARNING_RATE = 0.0003
EPOCHS = 40
LABEL_SMOOTHING = 0.01

data = DataHandling(max_train_rows=3000000, cv_frac=0.0)

train_ds, valid_ds = data.get_dataset()
cluster_list = data.clusters
cluster_cnt_list = data.clusters_cnt

del(data)
gc.collect()
memory_state()

model_sub = []
loss_object_sub = []
optimizer_sub = []
for idx_c, cluster in enumerate(cluster_list):
    model_sub.append(SubModel_Cluster(cluster, cluster_cnt_list[idx_c]))
    loss_object_sub.append(tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING))
    #optimizer_sub.append(tfa.optimizers.RectifiedAdam(learning_rate=LEARNING_RATE))
    optimizer_sub.append(tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
model_sub.append(MarketModel_A())
loss_object_sub.append(tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING))
optimizer_sub.append(tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

model = MarketModel_B(len(model_sub))
loss_object = tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.BinaryAccuracy(name='train_acc')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.BinaryAccuracy(name='test_acc')

@tf.function
def train_step(train_x, train_y):
    predictions_sub_list = []
    for i in range(len(model_sub)):
        with tf.GradientTape() as tape_sub:
            predictions_sub = model_sub[i](train_x, training=True)
            loss_sub = loss_object_sub[i](train_y, predictions_sub)
        gradients_sub = tape_sub.gradient(loss_sub, model_sub[i].trainable_variables)
        optimizer_sub[i].apply_gradients(zip(gradients_sub, model_sub[i].trainable_variables))
        predictions_sub = tf.math.reduce_mean(predictions_sub, axis=1)
        predictions_sub = tf.expand_dims(predictions_sub, axis=1)
        predictions_sub_list.append(predictions_sub)
        #train_loss(loss_sub)
    #predictions = sum(predictions_sub_list) / len(predictions_sub_list)
    with tf.GradientTape() as tape:
        predictions = model(predictions_sub_list, training=True)
        loss = loss_object(train_y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #tf.print(model.cluster_weights)
    #tf.print(predictions)
    train_loss(loss)
    train_acc.update_state(train_y[:, 0], predictions)

@tf.function
def valid_step(valid_x, valid_y):
    predictions_sub_list = []
    for i in range(len(model_sub)):
        predictions_sub = model_sub[i](valid_x, training=False)
        predictions_sub = tf.math.reduce_mean(predictions_sub, axis=1)
        predictions_sub = tf.expand_dims(predictions_sub, axis=1)
        predictions_sub_list.append(predictions_sub)
        #t_loss = loss_object(valid_y, predictions_sub)
        #test_loss(t_loss)
    #predictions = sum(predictions_sub_list) / len(predictions_sub_list)
    predictions = model(predictions_sub_list, training=False)
    t_loss = loss_object(valid_y, predictions)
    test_loss(t_loss)
    test_acc.update_state(valid_y[:, 0], predictions)

for epoch in range(EPOCHS):
    #pbar = tqdm(total=len(train_ds))
    start_time = time.time()
    for train_x, train_y in train_ds:
        train_step(train_x, train_y)
        #pbar.update(1)

    for valid_x, valid_y in valid_ds:
        valid_step(valid_x, valid_y)

    template = 'epoch {}, loss_train: ,{}, acc_train: ,{}, loss_test: ,{}, acc_test: ,{}, iter: {}s'
    print(template.format(epoch+1,
                           train_loss.result(),
                           train_acc.result()*100,
                           test_loss.result(),
                           test_acc.result()*100,
                           (time.time() - start_time)
                           ))
    #print(model.cluster_weights.numpy())

for idx, submodel in enumerate(model_sub):
    #submodel.summary()
    path = 'gs://blonix-tpu-bucket/kaggle_market/model_final/submodel_' + str(idx)
    tf.saved_model.save(submodel, path)
tf.saved_model.save(model, 'gs://blonix-tpu-bucket/kaggle_market/model_final/model')


# Inference Test

SUBMODEL_NUM = 5

loaded_submodel = []
for idx in range(SUBMODEL_NUM):
    path = "gs://blonix-tpu-bucket/kaggle_market/model_b/submodel_" + str(idx)
    loaded_submodel.append(tf.saved_model.load(path))
loaded_model = tf.saved_model.load("gs://blonix-tpu-bucket/kaggle_market/model_b/model")

infer_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/kaggle_market/datafiles/train.csv', nrows=10000, skiprows=list(range(1,1100000)))
infer_df.fillna(0, inplace=True)
features = [c for c in infer_df.columns if "feature" in c]

import math

class Evaluate():
    def __init__(self):
        self.p = 0
        self.p_tot = 0
        self.j = 0
        self.i = 1
        self.date_now = -1
        self.p_abs = 0
        self.corr = 0
        self.cnt = 0

    def calc(self, date, weight, resp, action):
        if self.date_now != date and self.p != 0:
            self.p_abs += math.sqrt(math.pow(self.p, 2))
            self.p_tot += self.p
            self.date_now = date
            self.i += 1
            self.p = 0
            self.j = 0
        if weight > 0:
            self.cnt += 1
            if (resp >= 0 and action == 1) or (resp < 0 and action == 0):
                self.corr += 1
        self.p += (weight * resp * action)
        self.j += 1

    def result(self):
        self.p_abs += math.sqrt(math.pow(self.p, 2))
        self.p_tot += self.p
        self.i += 1
        self.auc = (self.corr / self.cnt) * 100
        self.sharp = (self.p_tot / self.p_abs) * math.sqrt(250 / self.i)
        u = min(max(self.sharp, 0), 6) * self.p_tot
        return u


evaluate = Evaluate()

for index, row in tqdm(infer_df.iterrows()):
    action = 0
    test_np = row.to_numpy()
    test_np = np.expand_dims(test_np, axis=0)
    if test_np[0, 7] > 0:
        x_tt = test_np[:, 7:137].astype(np.float32)
        np.nan_to_num(x_tt, copy=False)
        # x_tt = scaler.transform(x_tt)

        pred_sub_list = []
        for i in range(SUBMODEL_NUM):
            pred_sub = loaded_submodel[i](tf.constant(x_tt), training=False)
            pred_sub = tf.math.reduce_mean(pred_sub, axis=1)
            pred_sub = tf.expand_dims(pred_sub, axis=1)
            pred_sub_list.append(pred_sub)
        pred = loaded_model(pred_sub_list, training=False)
        pred = np.median(pred.numpy())
        action = np.where(pred > 0.5, 1, 0).astype(int)
    else:
        action = 0
    evaluate.calc(row['date'], row['weight'], row['resp'], action)

score = evaluate.result()
template = 'p_sum:{}, t:{}, score:{}, auc:{}'
print(template.format(evaluate.p_tot, evaluate.sharp, score, evaluate.auc))
