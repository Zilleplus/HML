from tensorflow.keras.datasets import boston_housing
import tensorflow.keras as keras
import numpy as np

(train_data, train_targets), (test_data, test_targets) = \
        boston_housing.load_data()
# normalize the data to -> mean=0 std=1
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(units=64, activation="relu"),
        keras.layers.Dense(units=64, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mae", metrics=["mae"])
    return model


# k-fold cross validation
k = 4
num_val_samples = len(train_data)//k
num_epochs = 100
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1)*num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1)*num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        epochs=num_epochs,
                        validation_data=(val_data, val_targets),
                        batch_size=16, verbose=0)
    all_mae_histories.append(history.history)

print(all_mae_histories)

loss = np.mean([np.mean(hist["val_mae"]) for hist in all_mae_histories])
print(f"The mean average error is on average {loss:.2f}.")

# with 500 epochs -> 2.48 average
# with 100 epochs -> 3.08 average
