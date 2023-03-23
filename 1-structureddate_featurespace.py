import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.utils import FeatureSpace
from keras.utils.np_utils import to_categorical


###READ DATA###
header=['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways'
        ,'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'
        ,'Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'
        ,'Soil_Type1', 'Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8'
        ,'Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16'
        ,'Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24'
        ,'Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32'
        ,'Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40'
        ,'Cover_Type']

dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',compression='gzip',names=header)

dataframe=dataframe.iloc[:10000]

###Split Data####
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
#     % (len(train_dataframe), len(val_dataframe))
)

###Make Data Set Object###
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("Cover_Type")
    labels=to_categorical(labels)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

###PreProcess###
feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers
        "Slope": "integer_categorical",
        "Wilderness_Area1":"integer_categorical",
        "Wilderness_Area2":"integer_categorical",
        "Wilderness_Area3":"integer_categorical",
        "Wilderness_Area4":"integer_categorical",
        "Soil_Type1": "integer_categorical",
        "Soil_Type2": "integer_categorical",
        "Soil_Type3": "integer_categorical",
        "Soil_Type4": "integer_categorical",
        "Soil_Type5": "integer_categorical",
        "Soil_Type6": "integer_categorical",
        "Soil_Type7": "integer_categorical",
        "Soil_Type8": "integer_categorical",
        "Soil_Type9": "integer_categorical",
        "Soil_Type10": "integer_categorical",
        "Soil_Type11": "integer_categorical",
        "Soil_Type12": "integer_categorical",
        "Soil_Type13": "integer_categorical",
        "Soil_Type14": "integer_categorical",
        "Soil_Type15": "integer_categorical",
        "Soil_Type16": "integer_categorical",
        "Soil_Type17": "integer_categorical",
        "Soil_Type18": "integer_categorical",
        "Soil_Type19": "integer_categorical",
        "Soil_Type20": "integer_categorical",
        "Soil_Type21": "integer_categorical",
        "Soil_Type22": "integer_categorical",
        "Soil_Type23": "integer_categorical",
        "Soil_Type24": "integer_categorical",
        "Soil_Type25": "integer_categorical",
        "Soil_Type26": "integer_categorical",
        "Soil_Type27": "integer_categorical",
        "Soil_Type28": "integer_categorical",
        "Soil_Type29": "integer_categorical",
        "Soil_Type30": "integer_categorical",
        "Soil_Type31": "integer_categorical",
        "Soil_Type32": "integer_categorical",
        "Soil_Type33": "integer_categorical",
        "Soil_Type34": "integer_categorical",
        "Soil_Type35": "integer_categorical",
        "Soil_Type36": "integer_categorical",
        "Soil_Type37": "integer_categorical",
        "Soil_Type38": "integer_categorical",
        "Soil_Type39": "integer_categorical",
        "Soil_Type40": "integer_categorical",

        # Numerical features to normalize
        "Elevation": "float_normalized",
        "Aspect": "float_normalized",
        "Horizontal_Distance_To_Hydrology": "float_normalized",
        "Vertical_Distance_To_Hydrology": "float_normalized",
        "Horizontal_Distance_To_Roadways": "float_normalized",
        "Hillshade_9am": "float_normalized",
        "Hillshade_Noon": "float_normalized",
        "Hillshade_3pm": "float_normalized",
        "Horizontal_Distance_To_Fire_Points": "float_normalized",
    },
    # Our utility will one-hot encode all categorical
    # features and concat all features into a single
    # vector (one vector per sample).
    output_mode="concat",
)

train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print("preprocessed_x.shape:", preprocessed_x.shape)
    print("preprocessed_x.dtype:", preprocessed_x.dtype)

preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)


###Make Model and Fit###
dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(512, activation="relu")(encoded_features)
x = keras.layers.Dense(512, activation="relu")(x)
x = keras.layers.Dense(512, activation="relu")(x)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(8, activation="softmax")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)

training_model.fit(
    preprocessed_train_ds, epochs=20, validation_data=preprocessed_val_ds, verbose=2
)

"""
Epoch 1/20
250/250 - 4s - loss: 0.9547 - accuracy: 0.5971 - val_loss: 0.7293 - val_accuracy: 0.6905 - 4s/epoch - 15ms/step
Epoch 2/20
250/250 - 4s - loss: 0.7412 - accuracy: 0.6870 - val_loss: 0.6335 - val_accuracy: 0.7360 - 4s/epoch - 15ms/step
Epoch 3/20
250/250 - 3s - loss: 0.6905 - accuracy: 0.7056 - val_loss: 0.6243 - val_accuracy: 0.7315 - 3s/epoch - 13ms/step
Epoch 4/20
250/250 - 3s - loss: 0.6270 - accuracy: 0.7340 - val_loss: 0.5845 - val_accuracy: 0.7510 - 3s/epoch - 12ms/step
Epoch 5/20
250/250 - 4s - loss: 0.5899 - accuracy: 0.7494 - val_loss: 0.5677 - val_accuracy: 0.7525 - 4s/epoch - 16ms/step
Epoch 6/20
250/250 - 3s - loss: 0.5674 - accuracy: 0.7619 - val_loss: 0.5767 - val_accuracy: 0.7595 - 3s/epoch - 12ms/step
Epoch 7/20
250/250 - 3s - loss: 0.5381 - accuracy: 0.7820 - val_loss: 0.5558 - val_accuracy: 0.7765 - 3s/epoch - 12ms/step
Epoch 8/20
250/250 - 4s - loss: 0.5181 - accuracy: 0.7876 - val_loss: 0.5684 - val_accuracy: 0.7610 - 4s/epoch - 18ms/step
Epoch 9/20
250/250 - 3s - loss: 0.5123 - accuracy: 0.7935 - val_loss: 0.5595 - val_accuracy: 0.7710 - 3s/epoch - 12ms/step
Epoch 10/20
250/250 - 3s - loss: 0.4840 - accuracy: 0.7997 - val_loss: 0.5159 - val_accuracy: 0.7950 - 3s/epoch - 12ms/step
Epoch 11/20
250/250 - 5s - loss: 0.4544 - accuracy: 0.8094 - val_loss: 0.5599 - val_accuracy: 0.7650 - 5s/epoch - 18ms/step
Epoch 12/20
250/250 - 3s - loss: 0.4320 - accuracy: 0.8236 - val_loss: 0.5338 - val_accuracy: 0.7770 - 3s/epoch - 12ms/step
Epoch 13/20
250/250 - 3s - loss: 0.4215 - accuracy: 0.8271 - val_loss: 0.5845 - val_accuracy: 0.7805 - 3s/epoch - 13ms/step
Epoch 14/20
250/250 - 4s - loss: 0.3853 - accuracy: 0.8462 - val_loss: 0.5264 - val_accuracy: 0.7940 - 4s/epoch - 18ms/step
Epoch 15/20
250/250 - 3s - loss: 0.3730 - accuracy: 0.8484 - val_loss: 0.5537 - val_accuracy: 0.7770 - 3s/epoch - 12ms/step
Epoch 16/20
250/250 - 3s - loss: 0.3591 - accuracy: 0.8540 - val_loss: 0.6352 - val_accuracy: 0.7610 - 3s/epoch - 12ms/step
Epoch 17/20
250/250 - 5s - loss: 0.3474 - accuracy: 0.8568 - val_loss: 0.5683 - val_accuracy: 0.7870 - 5s/epoch - 18ms/step
Epoch 18/20
250/250 - 3s - loss: 0.3254 - accuracy: 0.8665 - val_loss: 0.5718 - val_accuracy: 0.7980 - 3s/epoch - 12ms/step
Epoch 19/20
250/250 - 3s - loss: 0.3158 - accuracy: 0.8700 - val_loss: 0.5569 - val_accuracy: 0.8025 - 3s/epoch - 12ms/step
Epoch 20/20
250/250 - 3s - loss: 0.2941 - accuracy: 0.8776 - val_loss: 0.5647 - val_accuracy: 0.8075 - 3s/epoch - 14ms/step
<keras.callbacks.History at 0x7f7883a68e20>
"""
