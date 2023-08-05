from ResUNet import *
from rectify import *
from test import *
from handle_images import *

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # cut images from UCL WebCam dataset to several 300 * 300 small images for further test predict
    cut_images("dataset/test/webcam/images", "dataset/test/webcam/cut_images")

    # Define image transformations

    image_simple_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224])
    ])

    image_weak_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
    ])

    image_strong_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(contrast=0.2),
    ])

    # Load training data
    train_dataset = CloudDataset("dataset/train/swinyseg/images", "dataset/train/swinyseg/GTmaps",
                                 transform=image_transforms)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Load validation data
    val_dataset = CloudDataset("dataset/val/swimseg/images", "dataset/val/swimseg/GTmaps", transform=image_transforms)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Load rectify data
    rectify_dataset = RectifyDataset("dataset/rectify/images", "dataset/rectify/GTmaps", transform=image_transforms)
    rectify_loader = DataLoader(rectify_dataset, batch_size=2, shuffle=True)

    transform_dataset = TransformDataset("dataset/rectify/images_", weak_transform=image_weak_transforms,
                                         strong_transform=image_strong_transforms)
    transform_loader = DataLoader(transform_dataset, batch_size=2, shuffle=True)

    # Load test data
    test_dataset = TestDataset("dataset/test/webcam/cut_images", transform=image_simple_transforms)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Check if a GPU is available and if not, use a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate model
    model = ResUNet(2)
    model.to(device)  # Move model to the specified device

    # Define loss function and optimizer
    criterion1 = nn.BCELoss()
    criterion2 = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # Train and Validate
    epochs = 5

    for epoch in range(epochs):

        model.train()
        train_iou = 0
        batch_iou = 0

        # training loop
        for images, masks, _, _ in train_loader:
            # Move data to specific device
            images = images.to(device)
            masks = masks.to(device)

            # Convert the mask to one-hot encoding
            masks = to_one_hot(masks, num_classes=2)

            # Forward pass
            outputs = model(images)
            loss = 0.5 * criterion1(outputs, masks) + 0.5 * criterion2(outputs, masks)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate train IoU
            iou, batch_iou = iou_score(outputs, masks)
            train_iou += iou

            print('Loss: {:.4f}, IoU: {:.4f}'.format(loss.item(), batch_iou))

        train_iou = train_iou / len(train_dataset)
        print('Epoch [{}/{}], Loss: {:.4f}, Train IoU: {:.4f}'.format(epoch + 1, epochs, loss.item(), train_iou))

        # validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_iou = 0
            count = 0
            test_count = 0
            val_f_score = 0
            for images, masks, origin_images, filenames in val_loader:
                result = masks[:, 0, :, :]
                images = images.to(device)
                masks = masks.to(device)

                # Convert the mask to one-hot encoding
                masks = to_one_hot(masks, num_classes=2)

                outputs = model(images)
                val_loss += 0.5 * criterion1(outputs, masks) + 0.5 * criterion2(outputs, masks)
                iou, batch_iou = iou_score(outputs, masks)
                fscore = f_score(outputs, masks)
                val_iou += iou
                val_f_score += fscore

                pred = torch.argmax(outputs, dim=1).cpu().numpy()

                # Save segmentation output
                for j in range(images.size(0)):
                    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
                    ax[0].imshow(origin_images[j].cpu())
                    ax[0].set_title('image')
                    ax[1].imshow(pred[j])
                    ax[1].set_title('predict')
                    ax[2].imshow(result[j])
                    ax[2].set_title('mask')
                    filename = os.path.splitext(filenames[j])[0]  # Remove the extension
                    plt.savefig(f'dataset/val/output/val_{filename}.png')
                    plt.close()
            print('Validation Loss: {:.4f}, Validation IoU: {:.2f}%, Validation F_score: {:.2f},'.format(
                val_loss / len(val_loader), val_iou * 100 / len(val_dataset), val_f_score * 2 / len(val_dataset)))
    torch.save(model, "ty_" + "train.pth")

    # Rectify
    epochs = 5
    for epoch in range(epochs):

        model.train()
        rectify_iou = 0
        batch_iou = 0

        for (images, masks), (weak_images, strong_images) in zip(rectify_loader, transform_loader):
            # Move data to specific device
            images = images.to(device)
            masks = masks.to(device)

            # Convert the mask to one-hot encoding
            masks = to_one_hot(masks, num_classes=2)

            # Forward pass
            outputs = model(images)
            loss1 = 0.5 * criterion1(outputs, masks) + 0.5 * criterion2(outputs, masks)

            weak_images = weak_images.to(device)
            strong_images = strong_images.to(device)

            outputs_ = model(weak_images)
            masks_ = outputs_.detach()

            strong_pre = model(strong_images)
            masks_[masks_ >= 0.8] = 1.0
            masks_[masks_ < 0.2] = 0.0
            loss2 = 0.5 * criterion1(strong_pre, masks_) + 0.5 * criterion2(strong_pre, masks_)

            loss = loss1 + loss2
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate train IoU
            iou, batch_iou = iou_score(outputs, masks)
            rectify_iou += iou

            print('Loss: {:.4f}, IoU: {:.4f}'.format(loss.item(), batch_iou))

        rectify_iou = rectify_iou / len(rectify_dataset)
        print('Epoch [{}/{}], Loss: {:.4f}, Rectify IoU: {:.4f}'.format(epoch + 1, epochs, loss.item(), rectify_iou))

    torch.save(model, "ty_" + "rectify.pth")

    # Test
    model.eval()
    with torch.no_grad():
        for images, origin_images, filenames in test_loader:

            images = images.to(device)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()

            # Save segmentation output
            for k in range(images.size(0)):
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(origin_images[k].cpu())
                ax[0].set_title('image')
                ax[1].imshow(pred[k])
                ax[1].set_title('predict')
                filename = os.path.splitext(filenames[k])[0]
                plt.savefig(f'dataset/test/output/test_{filename}.png')
                # Save the predicted mask as a separate image
                mask_img = Image.fromarray((pred[k] * 255).astype(np.uint8))
                mask_img.save(f'dataset/test/pred/pred_{filename}.jpg')
                plt.close()

    join_predicted_images("dataset/test/pred", "dataset/test/webcam/images/")
