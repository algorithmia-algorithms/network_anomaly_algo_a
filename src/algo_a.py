import Algorithmia
import tensorflow as tf
import shutil
from time import time

client = Algorithmia.client()

SM_PATH = "/tmp/saved_model"
MODEL_PATH = "data://network_anomaly_detection/models/modela_tf_23_aug2320.zip"

class TF_Logging():
    def __init__(self):
        self.events = []

    def insert_event(self, message):
        event = {"timestamp": str(time()), "message": message}
        self.events.append(event)


    def get_events(self):
        events = self.events
        self.events = []
        return events




def load():
    local_path = client.file(MODEL_PATH).getFile().name
    shutil.unpack_archive(local_path, SM_PATH, format='zip')
    model = tf.keras.models.load_model(SM_PATH, compile=False)
    return model


def apply(input):
    LOGGER = TF_Logging()
    LOGGER.insert_event("input to ALGO_A is: {}".format(input))
    LOGGER.insert_event("TF version: {}".format(tf.version.VERSION))
    tensor = tf.constant(input)
    outcome = MODEL.predict(tensor).tolist()[0][0]
    LOGGER.insert_event("outcome predicted by ALGO_A is: {}".format(outcome))
    output = {"outcome": outcome, "events": LOGGER.get_events()}
    return output

MODEL = load()

if __name__ == "__main__":
    output = apply([[4.0, 3.0, 2.2, -1.1, 0.25]])
    print(output)
    output = apply([[4.0, 3.0, 2.2, -1.1, 0.25]])
    print(output)
    output = apply([[4.0, 3.0, 2.2, -1.1, 0.25]])
    print(output)
