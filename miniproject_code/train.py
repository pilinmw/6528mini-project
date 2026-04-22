import os
import torch
import matplotlib
matplotlib.use('Agg')
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2



from dataset import SingleImageDataset
from model import CoordinateMLP

def train_implicit_representation(
    image_path,
    save_dir,
    experiment_name,
    target_height=256,
    num_frequencies=10,
    hidden_features=256,
    hidden_layers=4,
    epochs=1000,
    lr=1e-3,
    batch_size=8192
):
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{experiment_name}] Using device: {device}")
    
    # 1. Dataset
    dataset = SingleImageDataset(image_path, target_height=target_height)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    H, W = dataset.H, dataset.W
    
    # 2. Model
    model = CoordinateMLP(
        num_frequencies=num_frequencies, 
        hidden_features=hidden_features, 
        hidden_layers=hidden_layers
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 3. Training Loop
    losses = []
    
    print(f"[{experiment_name}] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for coords, colors in dataloader:
            coords, colors = coords.to(device), colors.to(device)
            
            optimizer.zero_grad()
            preds = model(coords)
            loss = criterion(preds, colors)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if epoch % 100 == 0 or epoch == 1:
            print(f"[{experiment_name}] Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}")
            
    # 4. Final Inference and Evaluation
    model.eval()
    with torch.no_grad():
        all_coords = dataset.coords.to(device)
        # Process in batches to avoid OOM
        preds_list = []
        for i in range(0, len(all_coords), batch_size):
            batch_coords = all_coords[i:i + batch_size]
            preds = model(batch_coords)
            preds_list.append(preds.cpu())
            
        final_img = torch.cat(preds_list, dim=0)
        final_img = final_img.view(H, W, 3).numpy()
        
    final_img_bgr = cv2.cvtColor((final_img * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
    save_path = os.path.join(save_dir, f"{experiment_name}_final.png")
    cv2.imwrite(save_path, final_img_bgr)
    
    # Calculate PSNR
    mse = np.mean((dataset.img_rgb.numpy() - final_img) ** 2)
    psnr = -10.0 * np.log10(mse) if mse > 0 else 100
    
    print(f"[{experiment_name}] Completed. PSNR: {psnr:.2f} dB, Saved to {save_path}")
    
    # Save loss plot
    plt.figure()
    plt.plot(losses)
    plt.title(f"{experiment_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_loss.png"))
    plt.close()
    
    return model, psnr, losses

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../jcsmr-1.jpg")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--baseline", action="store_true", help="Run without Positional Encoding")
    parser.add_argument("--epochs", type=int, default=500) # Fast test
    args = parser.parse_args()
    
    import numpy as np

    if args.baseline:
        train_implicit_representation(
            image_path=args.image,
            save_dir=args.outdir,
            experiment_name="baseline_no_pe",
            num_frequencies=0, # Abolition
            epochs=args.epochs
        )
    else:
        train_implicit_representation(
            image_path=args.image,
            save_dir=args.outdir,
            experiment_name="main_with_pe",
            num_frequencies=10, # With PE
            epochs=args.epochs
        )
