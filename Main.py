import torch
from Create_Dataset import RandomImagePixelationDataset
from sklearn.model_selection import train_test_split
from CNN import CNN
import matplotlib.pyplot as plt
from Training_Evaluation import training


def plot_losses(train_losses: list, eval_losses: list):
    for epoch, (train_loss, eval_loss) in enumerate(zip(train_losses, eval_losses), 1):
        print(f"Epoch {epoch}: Train Loss: {train_loss}, Eval Loss: {eval_loss}")
    print(f"lowest Train Loss: {min(train_losses)}")
    print(f"lowest eval Loss: {min(eval_losses)}")
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")     # for each epoch plot training_losses
    plt.plot(range(len(train_losses)), eval_losses, label="Evaluation Loss")    # for each epoch plot eval_losses
    plt.xlabel('Epochs')
    plt.ylabel('Root Mean-Squared-Error Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    torch.random.manual_seed(0)

    dataset = RandomImagePixelationDataset(
        image_dir="H:/Universität/Semester_2/Programming_in_Python_2/Assignment_7/training")
    print(len(dataset))
    # print(len(dataset)) #gives for dataset[0][0] pixelated image tensor, [0][1] known array,...

    train_data, test_data = train_test_split(dataset, test_size=0.1, shuffle=True)
    # creates Training and Test data sets (85/15)

    train_data, val_data = train_test_split(train_data, test_size=0.15, shuffle=False)
    # splits training data further into train and validation (85/15)
    # print(train_data[0][2].shape, train_data[0][2])
    print(f"Length Train_data: {len(train_data)}")
    print(f"Length Eval_Data: {len(val_data)}")
    print(f"Length Test_Data: {len(test_data)}")
    model = CNN(input_channels=1, hidden_channels=32, num_layers=5, num_av_pool=1, kernel_size=5)
    train_losses, eval_losses = training(model, train_data, val_data, test_data, num_epochs=30, learning_rate=0.01,
                                         show_progress=True, early_stop=False)
    print(eval_losses)
    plot_losses(train_losses, eval_losses)
