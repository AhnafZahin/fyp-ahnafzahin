import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import spearmanr, pearsonr
import cv2
from PIL import Image
import random
import glob
from tqdm import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset Paths
DATA_ROOT = "/content/drive/MyDrive/UIEB Dataset"
RAW_IMAGES_DIR = os.path.join(DATA_ROOT, "raw-890")
REF_IMAGES_DIR = os.path.join(DATA_ROOT, "reference-890")
PROCESSED_ROOT = "/content/drive/MyDrive/UIEB_processed"

# Validate dataset paths
def validate_dataset_paths():
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"Raw images directory not found: {RAW_IMAGES_DIR}")
        return False
    if not os.path.exists(REF_IMAGES_DIR):
        print(f"Reference images directory not found: {REF_IMAGES_DIR}")
        return False
    return True

# Create degradation directories
def create_degradation_directories():
    degradation_types = ['color_cast', 'blur', 'low_light', 'high_noise']
    for deg_type in degradation_types:
        os.makedirs(os.path.join(PROCESSED_ROOT, deg_type), exist_ok=True)
    return degradation_types

# Apply degradations
def apply_degradations_and_save():
    degradation_types = create_degradation_directories()
    all_img_paths = glob.glob(os.path.join(RAW_IMAGES_DIR, "**", "*.jpg"), recursive=True) + \
                    glob.glob(os.path.join(RAW_IMAGES_DIR, "**", "*.png"), recursive=True)

    print(f"Found {len(all_img_paths)} raw images. Applying degradations...")

    for img_path in tqdm(all_img_paths):
        img = cv2.imread(img_path)
        if img is None: continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(img_path)

        # Color Cast
        color_cast = img.copy()
        color_cast[:, :, 0] = np.clip(color_cast[:, :, 0] * 0.7, 0, 255)
        color_cast[:, :, 1] = np.clip(color_cast[:, :, 1] * 1.2, 0, 255)
        color_cast[:, :, 2] = np.clip(color_cast[:, :, 2] * 1.3, 0, 255)
        cv2.imwrite(os.path.join(PROCESSED_ROOT, 'color_cast', filename), color_cast[:, :, ::-1])

        # Blur
        blur = cv2.GaussianBlur(img, (15, 15), 5)
        cv2.imwrite(os.path.join(PROCESSED_ROOT, 'blur', filename), blur[:, :, ::-1])

        # Low Light
        low_light = np.clip(img * 0.4, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(PROCESSED_ROOT, 'low_light', filename), low_light[:, :, ::-1])

        # High Noise
        noise = img.copy()
        noise = np.clip(noise + np.random.normal(0, 25, noise.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(PROCESSED_ROOT, 'high_noise', filename), noise[:, :, ::-1])

    print("Finished applying degradations.")

# Attention Fusion Module
class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
    def forward(self, raw_features, ref_features):
        combined = torch.cat([raw_features, ref_features], dim=1)
        attention_map = self.attention(combined)
        attended_raw = raw_features * attention_map
        fused = attended_raw + ref_features
        return fused

# Improved Dual-Stream Model with ResNet Backbone
class NovelUnderwaterImageAssessmentModel(nn.Module):
    def __init__(self, num_classes=4):
        super(NovelUnderwaterImageAssessmentModel, self).__init__()
        
        # Pretrained ResNet backbones
        self.raw_stream = models.resnet18(pretrained=True)
        self.raw_stream = nn.Sequential(*list(self.raw_stream.children())[:-2])
        
        self.ref_stream = models.resnet18(pretrained=True)
        self.ref_stream = nn.Sequential(*list(self.ref_stream.children())[:-2])

        # Attention fusion
        self.fusion = AttentionFusion(512)
        
        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Improved Quality Head
        self.quality_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Degradation Head with Label Smoothing
        self.degradation_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.fusion.apply(init_fn)
        self.shared.apply(init_fn)
        self.quality_head.apply(init_fn)
        self.degradation_head.apply(init_fn)

    def forward(self, raw_img, ref_img=None):
        raw_features = self.raw_stream(raw_img)
        
        if ref_img is not None:
            ref_features = self.ref_stream(ref_img)
            fused_features = self.fusion(raw_features, ref_features)
        else:
            fused_features = raw_features
        
        shared_features = self.shared(fused_features)
        quality = self.quality_head(shared_features)
        degradation = self.degradation_head(shared_features)
        
        # Ensure proper output shapes
        quality = quality.view(-1)  # Flatten to [batch_size]
        return quality, degradation

# Dataset Class with Augmentation
class PairedUnderwaterImageDataset(Dataset):
    def __init__(self, raw_dir, ref_dir, processed_dir, transform=None, is_train=True):
        self.raw_dir = raw_dir
        self.ref_dir = ref_dir
        self.processed_dir = processed_dir
        self.transform = transform
        self.is_train = is_train

        self.ref_files = [f for f in os.listdir(ref_dir) if f.endswith(('.jpg', '.png'))]
        self.valid_files = []
        for ref_file in self.ref_files:
            raw_path = os.path.join(raw_dir, ref_file)
            if os.path.exists(raw_path):
                self.valid_files.append(ref_file)

        # Initialize data storage
        self.image_paths = []
        self.labels = []
        self.quality_scores = []
        self.ref_image_paths = []

        degradation_types = ['color_cast', 'blur', 'low_light', 'high_noise']
        quality_mapping = {'color_cast': 0.7, 'blur': 0.5, 'low_light': 0.4, 'high_noise': 0.3}

        # Add original images
        for filename in self.valid_files:
            raw_path = os.path.join(raw_dir, filename)
            self.image_paths.append(raw_path)
            self.labels.append(0)
            self.quality_scores.append(0.85)
            self.ref_image_paths.append(os.path.join(ref_dir, filename))

        # Add degraded images
        for label, deg_type in enumerate(degradation_types):
            deg_dir = os.path.join(processed_dir, deg_type)
            if not os.path.exists(deg_dir): continue
                
            img_paths = glob.glob(os.path.join(deg_dir, "*.jpg")) + glob.glob(os.path.join(deg_dir, "*.png"))
            for img_path in img_paths:
                filename = os.path.basename(img_path)
                ref_path = os.path.join(ref_dir, filename)
                if os.path.exists(ref_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    self.quality_scores.append(quality_mapping[deg_type])
                    self.ref_image_paths.append(ref_path)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        ref_path = self.ref_image_paths[idx]
        
        image = Image.open(img_path).convert('RGB')
        ref_image = Image.open(ref_path).convert('RGB')

        if self.transform:
            if self.is_train and random.random() > 0.5:
                image = transforms.functional.hflip(image)
                ref_image = transforms.functional.hflip(ref_image)
            image = self.transform(image)
            ref_image = self.transform(ref_image)

        quality_score = torch.tensor(self.quality_scores[idx], dtype=torch.float32)
        degradation_type = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, ref_image, quality_score, degradation_type

# Enhanced Training Function
def train_model(model, train_loader, val_loader, num_epochs=20):
    writer = SummaryWriter()
    quality_criterion = nn.MSELoss()
    
    # Class weights for imbalanced data
    class_counts = np.bincount(train_loader.dataset.dataset.labels)
    class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-6))
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    degradation_criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1
    )

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    warmup_scheduler, reduce_scheduler = get_lr_schedulers(optimizer)
    
    history = {'train_loss': [], 'val_loss': [], 'val_quality_corr': [], 'val_degradation_acc': []}
    best_val_loss = float('inf')
    no_improve_epochs = 0
    quality_weight = 0.8
    degradation_weight = 0.2

    for epoch in range(num_epochs):
        # Initialize epoch tracking
        model.train()
        train_loss = 0.0
        total_samples = 0
        
        # Create outer progress bar for epoch
        epoch_bar = tqdm(range(len(train_loader)), desc=f'Epoch {epoch+1}/{num_epochs}', position=0, leave=True)
        
        for i, (images, ref_images, quality_scores, degradation_types) in enumerate(train_loader):
            # Skip if batch size is 1
            if images.size(0) == 1:
                continue
                
            images = images.to(device)
            ref_images = ref_images.to(device)
            quality_scores = quality_scores.to(device)
            degradation_types = degradation_types.to(device)

            # Forward pass
            pred_quality, pred_degradation = model(images, ref_images)
            
            # Ensure proper shapes
            pred_quality = pred_quality.view(-1)
            quality_scores = quality_scores.view(-1)

            # Calculate losses
            quality_loss = quality_criterion(pred_quality, quality_scores)
            degradation_loss = degradation_criterion(pred_degradation, degradation_types)
            loss = quality_weight * quality_loss + degradation_weight * degradation_loss

            # Backward pass and optimize
            loss.backward()
            
            if (i+1) % 4 == 0:  # Gradient accumulation steps
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Update learning rate after optimizer step
                if epoch < 5:
                    warmup_scheduler.step()

            # Update metrics
            train_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Update progress bar
            epoch_bar.set_postfix({
                'loss': f"{train_loss/total_samples:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            epoch_bar.update()

        # Close epoch bar
        epoch_bar.close()

        # Validation
        train_loss /= total_samples
        val_loss, pearson_corr, degradation_acc = validate(model, val_loader, quality_criterion, degradation_criterion)
        
        # Update learning rate scheduler based on validation loss
        reduce_scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_quality_corr'].append(pearson_corr)
        history['val_degradation_acc'].append(degradation_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= 7:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Quality Pearson: {pearson_corr:.4f} | Degradation Acc: {degradation_acc:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}\n")

        # TensorBoard logging
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('Metrics/Quality_Corr', pearson_corr, epoch)
        writer.add_scalar('Metrics/Degradation_Acc', degradation_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

    writer.close()
    return history

def validate(model, val_loader, quality_criterion, degradation_criterion):
    model.eval()
    val_loss = 0.0
    pred_quality, true_quality = [], []
    pred_degradation, true_degradation = [], []

    with torch.no_grad():
        for images, ref_images, q_scores, d_types in val_loader:
            images, ref_images = images.to(device), ref_images.to(device)
            q_scores, d_types = q_scores.to(device), d_types.to(device)

            q_pred, d_pred = model(images, ref_images)
            q_loss = quality_criterion(q_pred, q_scores)
            d_loss = degradation_criterion(d_pred, d_types)
            loss = 0.8 * q_loss + 0.2 * d_loss

            val_loss += loss.item() * images.size(0)
            pred_quality.extend(q_pred.cpu().numpy())
            true_quality.extend(q_scores.cpu().numpy())
            pred_degradation.extend(torch.argmax(d_pred, 1).cpu().numpy())
            true_degradation.extend(d_types.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    pearson_corr, _ = pearsonr(true_quality, pred_quality)
    acc = accuracy_score(true_degradation, pred_degradation)
    return val_loss, pearson_corr, acc

def get_lr_schedulers(optimizer):
    warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e+1)/5))
    reducer = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    return warmup, reducer

# Main Training Execution
def run_training():
    if not validate_dataset_paths():
        print("Invalid dataset paths")
        return

    if not os.path.exists(PROCESSED_ROOT):
        apply_degradations_and_save()

    # Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    full_dataset = PairedUnderwaterImageDataset(
        RAW_IMAGES_DIR, REF_IMAGES_DIR, PROCESSED_ROOT, 
        transform=train_transform, is_train=True
    )

    # Split dataset
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=full_dataset.labels
    )

      # In run_training() function, modify dataset splitting:
    train_sub = torch.utils.data.Subset(full_dataset, train_idx)
    val_sub = torch.utils.data.Subset(full_dataset, val_idx)

    # Apply transforms separately
    train_sub.dataset.transform = train_transform
    val_sub.dataset.transform = val_transform
    # Data loaders
    train_loader = DataLoader(train_sub, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, drop_last = True)  # Reduced workers
    val_loader = DataLoader(val_sub, batch_size=8, shuffle=False, num_workers=0, pin_memory=False, drop_last = True)
    
    # Initialize model
    model = NovelUnderwaterImageAssessmentModel().to(device)
    print(model)

    # Train
    history = train_model(model, train_loader, val_loader, num_epochs=10)

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(history['val_quality_corr'], label='Quality Corr')
    plt.plot(history['val_degradation_acc'], label='Degradation Acc')
    plt.title('Metrics')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model

# Run training
if __name__ == "__main__":
    trained_model = run_training()
