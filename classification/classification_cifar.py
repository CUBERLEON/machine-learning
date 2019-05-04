import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from classification.models.cifar import custom_cifar_medium_model
from common.config import REMOTE_DATA_URL, KERAS_DIR
from common.plot import plot_curves, plot_confusion_matrix, plot_cnn_layer
from common.utils import download, extract
from datasets.cifar_dataset import load_data


if __name__ == '__main__':
    data_dir = extract(download(REMOTE_DATA_URL / "cifar" / "cifar100superclass.zip", KERAS_DIR / "cifar"))

    # class_names = ['shark', 'sunflower', 'lizard', 'motorcycle', 'bus']  # 1
    class_names = ['shark', 'sunflower', 'bed', 'motorcycle', 'palm_tree']  # 2
    labels = {label: i for i, label in enumerate(class_names)}
    img_shape = (32, 32)
    num_classes = len(labels)

    x_all, y_all = load_data(data_dir / "train", labels, img_shape)
    print(x_all.shape, y_all.shape)
    x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.1)

    x_test, y_test = load_data(data_dir / "test", labels, img_shape)

    model = custom_cifar_medium_model(x_train.shape[1:], num_classes)
    print(model.summary())

    history = model.fit(x_train, y_train, batch_size=250, epochs=100, validation_data=(x_val, y_val), shuffle=True)

    plot_curves([history.history['acc'], history.history['val_acc']], title='Model accuracy',
                xlabel='Epoch', ylabel='Accuracy', legend=['Train', 'Validation'])
    plot_curves([history.history['loss'], history.history['val_loss']], title='Model loss',
                xlabel='Epoch', ylabel='Loss', legend=['Train', 'Validation'])

    y_pred = model.predict_classes(x_test)
    y_pred_proba = model.predict_proba(x_test)

    print(classification_report(y_test, y_pred, target_names=class_names))

    for i in np.random.choice(len(x_test), 5):
        img = (x_test[i].reshape(*img_shape, 3) * 255).astype(int)

        plt.imshow(img)
        d = [f"\n{class_name}: {y_pred_proba[i][k]:.3f}" for k, class_name in enumerate(class_names)]
        plt.xlabel(f"label = {class_names[int(y_test[i])]}, predict = {class_names[int(y_pred[i])]}:{''.join(d)}")
        plt.show()

    plot_confusion_matrix(y_test, y_pred, classes=np.array(class_names),
                          title='Confusion matrix')

    plot_confusion_matrix(y_test, y_pred, classes=np.array(class_names), normalize=True,
                          title='Normalized confusion matrix')

    test_img = x_test[70]
    plt.imshow(test_img)
    plt.title('Test image used for layers visualization')
    plt.show()
    for name in ['conv_1', 'conv_2', 'conv_3']:
        plot_cnn_layer(model, test_img=test_img, layer=model.get_layer(name=name), title=name)
