import os
import logging
import torch
from torch.utils.data import DataLoader
from src.models import HybridRoofModel, RoofLoss
from src.data import RoofDataset
from src.prepare_training import setup_training
from src.train import train_model
from src.result_manager import ResultManager

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    try:
        # Setup training configuration and data splits
        logging.info("Setting up training configuration...")
        config, train_subset, val_subset = setup_training(
            train_size=4,    # Using 4 samples for training
            val_size=1       # Using 1 sample for validation
        )
        
        # Initialize result manager
        result_manager = ResultManager(config.results_dir)
        
        # Create datasets
        logging.info("Creating datasets...")
        train_dataset = RoofDataset(train_subset, list(range(len(train_subset))))
        val_dataset = RoofDataset(val_subset, list(range(len(val_subset))))
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=2,  # Process 2 images at a time
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Process 1 validation image at a time
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model and training components
        logging.info("Initializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HybridRoofModel(num_segment_classes=config.num_segment_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = RoofLoss()
        
        # Train model
        logging.info(f"Starting training on {device}")
        logging.info(f"Training samples: {len(train_subset)}")
        logging.info(f"Validation samples: {len(val_subset)}")
        logging.info(f"Batch size: {config.batch_size}")
        logging.info(f"Number of epochs: {config.num_epochs}")
        logging.info(f"Results will be saved in: {result_manager.run_dir}")
        
        model, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            result_manager=result_manager
        )
        
        logging.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        logging.info(f"Results saved in: {result_manager.run_dir}")
        
        # Print paths to important results
        print("\nResults Summary:")
        print(f"- Training log: {os.path.join(result_manager.run_dir, 'logs', 'training.log')}")
        print(f"- Best model: {os.path.join(result_manager.run_dir, 'checkpoints', 'best.pth')}")
        print(f"- Training report: {os.path.join(result_manager.run_dir, 'report', 'index.html')}")
        print(f"- Loss plot: {os.path.join(result_manager.run_dir, 'metrics', 'loss_plot.png')}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
