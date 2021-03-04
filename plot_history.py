from matplotlib.pyplot import clf
import matplotlib.pyplot as plt


def plot_history(net_history):
    history = net_history.history
    losses = history['loss']
    val_losses = history['val_loss']
    # accuracies = history['accuracy'] if accuracy else None
    # val_accuracies = history['val_accuracy'] if accuracy else None

    clf()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['train_loss', 'val_loss'])

    # plt.figure()
    # clf()
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.plot(accuracies)
    # plt.plot(val_accuracies)
    # plt.legend(['accuracy', 'val_accuracy'])
