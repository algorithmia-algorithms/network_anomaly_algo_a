import Algorithmia
import tensorflow as tf
import shutil

client = Algorithmia.client()


sm_path = "/tmp/saved_model"
model_path = "data://network_anomaly_detection/models/modela_tf_23_aug2320.zip"

def load():
    local_path = client.file(model_path).getFile().name
    shutil.unpack_archive(local_path, sm_path, format='zip')
    model = tf.keras.models.load_model(sm_path, compile=False)
    return model


def apply(input):
    tensor = tf.constant(input)
    output = model.predict(tensor)
    return output

model = load()


output = apply([[4.0, 3.0, 2.2, -1., 0.25]])
print(output)