import keras
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from classification.models.cifar import cifar_model_builder, cifar_model_builder_inv
from common.config import REMOTE_DATA_URL, KERAS_DIR, MODELS_DATA_DIR
from common.plot import plot_confusion_matrix, plot_pca_2d, plot_tsne_2d, plot_cnn_layer
from common.utils import download, extract, keras_model, retrieve_name, layer_functor
from datasets.cifar_dataset import load_data


if __name__ == '__main__':
    np.random.seed(25)

    classes_close = ['train', 'bicycle', 'tank', 'motorcycle', 'bus']
    classes_far = ['shark', 'sunflower', 'bed', 'motorcycle', 'palm_tree']

    train_all = False
    model_subtype = "medium"
    class_names = classes_far

    data_dir = extract(download(REMOTE_DATA_URL / "cifar" / "cifar100superclass.zip", KERAS_DIR / "cifar"))

    labels = {label: i for i, label in enumerate(class_names)}
    img_shape = (32, 32)
    num_classes = len(labels)

    x_train, y_train = load_data(data_dir / "train", labels, img_shape)
    x_test, y_test = load_data(data_dir / "test", labels, img_shape)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    model = cifar_model_builder(model_subtype)(x_train.shape[1:], num_classes)
    print(model.summary())

    keras_model(model,
                MODELS_DATA_DIR / f'custom_cifar_{model_subtype}_{len(class_names)}_{retrieve_name(class_names)}',
                train=train_all, train_data=train_test_split(x_train, y_train, test_size=0.1))

    y_pred = model.predict_classes(x_test)
    y_pred_proba = model.predict_proba(x_test)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # confusion matrices
    plot_confusion_matrix(y_test, y_pred, classes=np.array(class_names), normalize=True,
                          title='Normalized confusion matrix')

    # cnn filters
    test_img = x_test[70]
    plt.imshow(test_img)
    plt.title('Test image used for layers visualization')
    plt.show()
    # for i in range(9):
    #     plot_cnn_layer(model, test_img=test_img, layer=model.get_layer(index=i), title=f"Layer #{i}")
    convs = ['conv_1']
    if model_subtype == 'small' or model_subtype == 'medium':
        convs.append('conv_2')
    if model_subtype == 'medium':
        convs.append('conv_3')
    for name in convs:
        plot_cnn_layer(model, test_img=test_img, layer=model.get_layer(name=name), title=name, normalize=False)

    # features visualization and deconvolutions
    deconv_model = cifar_model_builder_inv(model_subtype)(None, None)
    deconv_model.summary()

    features_train = layer_functor(model, 'features')(x_train)

    keras_model(deconv_model,
                MODELS_DATA_DIR / f'custom_cifar_{model_subtype}_{len(class_names)}_{retrieve_name(class_names)}_deconv',
                train=train_all, train_data=train_test_split(features_train, x_train, test_size=0.1))

    restored = deconv_model.predict(features_train)
    for i in np.random.randint(0, len(features_train), 3):
        fig, (ax_orig, ax_rest) = plt.subplots(ncols=2)
        ax_orig.imshow(x_train[i])
        ax_orig.set_title('original')
        ax_rest.imshow(restored[i])
        ax_rest.set_title('restored')
        plt.show()

    fig, (ax_pca, ax_tsne) = plt.subplots(ncols=2)
    plot_pca_2d(features_train, y_train, ax=ax_pca, class_names=class_names)
    plot_tsne_2d(features_train, y_train, ax=ax_tsne, class_names=class_names, verbose=True)
    plt.show()
