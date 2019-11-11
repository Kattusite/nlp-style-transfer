# see https://www.tensorflow.org/guide/keras/save_and_serialize

# several imports will be needed
from tensorflow import keras
import load_data
import vectorize_data

def load_model(model_file):
    """Load the keras model from the given file and return it"""

    new_model = keras.models.load_model(model_file)

    return new_model


def predict(model, test_data):
    """ predict the results for the test dataset given the trained model"""


    pred = model.predict(test_data)

    print(pred)


    # np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

def main():
    model = load_model("dickens_verne_sepcnn_model.h5")

    dev, dlabels = load_data.load_data("data/merged/dev.txt")

    test, tlabels = load_data.load_data("data/merged/test.txt")

    vtest, vdev, wids = vectorize_data.sequence_vectorize(dev, test)

    preds = model.predict(vtest)

    authors = [
        "Verne",
        "Dickens"
    ]

    for i, t in enumerate(test):
        p = preds[i][0]
        pred = int(round(p))
        p_auth = authors[pred]
        a_auth = authors[tlabels[i]]
        s = "%s   %.4f ==> %s (actual: %s)" % (t, p, p_auth, a_auth)

        if p_auth != a_auth:
            print(s)

        if i > 350:
            break



if __name__ == '__main__':
    main()
