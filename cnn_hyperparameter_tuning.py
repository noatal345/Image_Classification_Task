import wandb
from main import img_classifier

wandb.login()


def main():
    wandb.init()
    test_acc = img_classifier(wandb.config)
    wandb.log({"Test accuracy": test_acc})


sweep_config = {
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'Test accuracy'},
    'parameters':
        {
            'lr': {'values': [0.00001, 0.0001, 0.001, 0.01]},
            'epoch': {'values': [10, 20, 30]},
            'batch_size': {'values': [16, 32, 64]},
        }
}

# Create a sweep
sweep_id = wandb.sweep(sweep=sweep_config, project='Image_classification_task')
# Run the sweep
wandb.agent(sweep_id, function=main)
