import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from Stack_function import stack_with_padding
from PIL import Image
from torchvision import transforms


def rmse_loss(predictions, targets):
    mse_loss = torch.nn.MSELoss()
    mse = mse_loss(predictions, targets)
    rmse = torch.sqrt(mse)
    return rmse


def training(
        model: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        test_data: torch.utils.data.Dataset,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        show_progress: bool = False,
        early_stop: bool = True
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device=device)
    train_loss = []
    eval_loss = []
    test_loss = []
    batch_size = batch_size

    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True, collate_fn=stack_with_padding)
    eval_loader = DataLoader(eval_data, batch_size, shuffle=False, drop_last=True, collate_fn=stack_with_padding)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True, collate_fn=stack_with_padding)

    # print(train_loader)
    min_eval_loss = float('inf')
    early_stop_count = 0
    model.train()    # sets model to training mode
    for epochs in range(num_epochs):    # iterate over the number of epochs
        batch_loss_training = []     # to store each batch loss
        if show_progress:   # for progress bar
            train_loader = tqdm.tqdm(train_loader, desc=f"Training Epoch {epochs + 1}")
        for inputs, known_array, targets in train_loader:    # iterate over the train_loader for each individual batch
            inputs = inputs.float().to(device=device)
            targets = targets.float().to(device=device)

            optimizer.zero_grad()   # sets the gradients to zero again that they don't accumulate

            output = model(inputs)  # invoke the forward method of the network with input

            #print(f"input shape: {inputs.shape}")
            #print(f"target shape: {targets.shape}")
            #print(f"output shape: {output.shape}")
            # targets = targets.view(output.shape)

            modified_output = output[known_array]
            modified_targets = targets[known_array]
            # print(f"modified_output: {modified_output}")
            # print(f"modified_targets: {modified_targets}")
            loss = rmse_loss(modified_output, modified_targets)
            loss.backward()     # backward pass
            optimizer.step()    # update weights
            batch_loss_training.append(loss.item())  # appends the loss per each batch to the list
        train_loss.append(sum(batch_loss_training)/len(batch_loss_training))  # summing up each individual batch loss and divide it by
        # batch count to get the accumulated loss for each epoch
        print(f"Train_Loss: {train_loss[-1]}")

        model.eval()    # puts model in evaluation mode
        batch_loss_eval = []
        for inputs, known_array, targets in eval_loader:
            with torch.no_grad():   # disables gradient tracking
                inputs = inputs.float().to(device=device)
                targets = targets.float().to(device=device)
                output = model(inputs)
                # print(inputs.shape)
                # print(targets.shape)
                # print(output.shape)
                # targets = targets.view(output.shape)
                modified_output = output[known_array]
                modified_targets = targets[known_array]
                # print(f"modified_output: {modified_output}")
                # print(f"modified_targets: {modified_targets}")
                loss = rmse_loss(modified_output, modified_targets)
                batch_loss_eval.append(loss.item())
            """
            for batch in targets:
                for img in batch:
                    img_1 = transforms.ToPILImage()(img)
                    img_1.show()
                    break

            for batch in output:
                for img in batch:
                    img_2 = transforms.ToPILImage()(img)
                    img_2.show()
                    break
            """
        eval_loss.append(sum(batch_loss_eval)/len(batch_loss_eval))
        print(f"Eval_Loss: {eval_loss[-1]}")

        if early_stop:
            if eval_loss[-1] < min_eval_loss:
                min_eval_loss = eval_loss[-1]
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count == 5:
                break

    model.eval()  # puts model in evaluation mode
    batch_loss_test = []
    for inputs, known_array, targets in test_loader:
        with torch.no_grad():  # disables gradient tracking
            inputs = inputs.float().to(device=device)
            targets = targets.float().to(device=device)
            output = model(inputs)
            #print(inputs.shape)
            #print(targets.shape)
            #print(output.shape)
            # targets = targets.view(output.shape)
            modified_output = output[known_array]
            modified_targets = targets[known_array]
            # print(f"modified_output: {modified_output}")
            # print(f"modified_targets: {modified_targets}")
            loss = rmse_loss(modified_output, modified_targets)
            batch_loss_test.append(loss.item())
        for batch in inputs:
            for img in batch:
                img_1 = transforms.ToPILImage()(img)
                break
            break
        for batch in targets:
            for imag in batch:
                img_2 = transforms.ToPILImage()(imag)
                break
            break

        for batch in output:
            for image in batch:
                img_3 = transforms.ToPILImage()(image)
                break
            break
        break
    test_loss.append(sum(batch_loss_test) / len(batch_loss_test))
    # print(f"Test_Loss: {test_loss[-1]}")

    return train_loss, eval_loss, test_loss, img_1, img_2, img_3
