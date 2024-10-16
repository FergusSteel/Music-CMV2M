import argparse
from preprocess import preprocess_data
from prototype1.model1 import VisualAudioModel

# Fergus Steel (2542391s) MSCi Project, this is a basic main function that will be used to run the model

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Cross-modal Audio-Visual Translation")
    parser.add_argument('--train', action='store_true', help="Training Script")
    parser.add_argument('--evaluate', action='store_true', help="Evaluaion Script")
    parser.add_argument('--dataset', type=str, default='dataset/', help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    args = parser.parse_args()

    # Preprocess the data
    train_loader, val_loader = preprocess_data(args.dataset, args.batch_size)

    # Initialize the model
    model = VisualAudioModel()

    if args.train:
        print(train_loader)
        print(args.epochs)
        model.train_model(train_loader=train_loader, epochs=args.epochs)
    
    if args.evaluate:
        model.evaluate(val_loader)

if __name__ == "__main__":
    main()
