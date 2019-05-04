import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from classification.models.cifar import custom_cifar_gru_model, custom_cifar_lstm_model
from common.config import REMOTE_DATA_URL, KERAS_DIR, MODELS_DATA_DIR
from common.plot import plot_confusion_matrix
from common.utils import download, extract, keras_model
from datasets.cifar_dataset import load_data


if __name__ == '__main__':
    data_dir = extract(download(REMOTE_DATA_URL / "cifar" / "cifar100superclass.zip", KERAS_DIR / "cifar"))

    # class_names = ['shark', 'sunflower', 'lizard', 'motorcycle', 'bus']  # 1
    class_names = ['shark', 'sunflower', 'bed', 'motorcycle', 'palm_tree']  # 2
    labels = {label: i for i, label in enumerate(class_names)}
    img_size = 32
    num_classes = len(labels)

    x_train, y_train = load_data(data_dir / "train", labels, (img_size, img_size))
    x_test, y_test = load_data(data_dir / "test", labels, (img_size, img_size))
    x_train = np.reshape(x_train, (-1, img_size, img_size * 3))
    x_test = np.reshape(x_test, (-1, img_size, img_size * 3))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    model = custom_cifar_lstm_model(x_train.shape[1:], num_classes)
    print(model.summary())

    keras_model(model, MODELS_DATA_DIR / 'custom_cifar_lstm_5_2_11111',
                train=True, train_data=train_test_split(x_train, y_train, test_size=0.1, random_state=42))

    y_pred = model.predict_classes(x_test)
    y_pred_proba = model.predict_proba(x_test)

    print(classification_report(y_test, y_pred, target_names=class_names))

    for i in np.random.choice(len(x_test), 5):
        img = (x_test[i].reshape(img_size, img_size, 3) * 255).astype(int)

        plt.imshow(img)
        d = [f"\n{class_name}: {y_pred_proba[i][k]:.3f}" for k, class_name in enumerate(class_names)]
        plt.xlabel(f"label = {class_names[int(y_test[i])]}, predict = {class_names[int(y_pred[i])]}:{''.join(d)}")
        plt.show()

    plot_confusion_matrix(y_test, y_pred, classes=np.array(class_names), normalize=True, title='Сonfusion matrix')
