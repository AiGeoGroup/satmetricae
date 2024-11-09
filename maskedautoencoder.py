import os

import torch
import torchvision
import lightly.data as data
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform
from timm.models.vision_transformer import vit_base_patch32_224
from torch import nn


IMAGENET_NORMALIZE = {}
IMAGENET_NORMALIZE["mean"], IMAGENET_NORMALIZE["std"] = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # 0-255 归一 0-1


class MAE(nn.Module):
    def __init__(self, vit, decoder_dim=512, mask_ratio=0.75):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = vit.patch_embed.patch_size[0]

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(self.decoder.mask_token,
                                      (batch_size, self.sequence_length))
        x_masked = utils.set_at_index(x_masked, idx_keep,
                                      x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(x_encoded=x_encoded,
                                      idx_keep=idx_keep,
                                      idx_mask=idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        return x_pred, target


def get_dataloader_ssl(
        configs,
        train_data_root='/kaggle/input/siri-whu-train-test-dataset/Dataset/train/',
        test_data_root='/kaggle/input/siri-whu-train-test-dataset/Dataset/test/'):

    preprocess = MAETransform()

    def target_transform(t): return 0


    train_dataset = torchvision.datasets.ImageFolder(
        train_data_root,
        transform=preprocess,
        target_transform=target_transform,
    )

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (configs['input_size'], configs['input_size'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_NORMALIZE["mean"],
            std=IMAGENET_NORMALIZE["std"],
        ),
    ])

    # create a lightly dataset for embedding
    dataset_test = data.LightlyDataset(input_dir=test_data_root,
                                       transform=test_transforms)

    # create a dataloader for embedding
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=configs['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    return dataloader_train, dataloader_test


def train_MAE(dataloader, train_iter=10, dur_times=5):
    vit = vit_base_patch32_224()
    model = MAE(vit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

    print("Starting Training")
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            views = batch[0]
            images = views[0].to(device)  # views contains only a single view
            predictions, targets = model(images)
            loss = criterion(predictions, targets)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)

        if epoch % dur_times == 0:
            print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    return model


def save_model_state_dict(
        mae_model,
        save_model_to_path='/kaggle/working/model_state_dict_files/',
        save_model_names='mae_model_scripted_',
        epochs=10,
        mask_ratio=0.75):

    if not os.path.exists(save_model_to_path):
        os.makedirs(save_model_to_path)

    save_model_names = save_model_names + 'ae_model_scripted_' + str(
        epochs) + str(mask_ratio) + '.pt'

    torch.save(mae_model.state_dict(), save_model_to_path + save_model_names)


def load_model_state_dict(
        mae_model,
        mdoels_PATH='/kaggle/input/model_state_dict_files/',
        save_model_names='mae_model_scripted_',
        epochs=10,
        mask_ratio=0.75):
    save_model_names = save_model_names + 'ae_model_scripted_' + str(
        epochs) + str(mask_ratio) + '.pt'

    torch.save(mae_model.state_dict(), save_model_to_path + save_model_names)
    mae_model.load_state_dict(
        torch.load(mdoels_PATH + 'ae_model_scripted_' + str(epochs) + '.pt',
                   weights_only=False))
    return mae_model
