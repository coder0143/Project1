import json
import os
import traceback
     
class TorchTrack:
    def __init__(self, experiment_name='default'):
        """
        Initialize TorchTrack with an experiment name.
        
        Args:
            experiment_name (str): Name of the current experiment
        """
        self.data_file = 'data.json'
        self.experiment_name = experiment_name
        self.epoch_data = {
            'loss_per_epoch': [],
            'accuracy_per_epoch': []
        }
        
        # Ensure data.json exists and is initialized
        self._initialize_data_file()
    
    def _initialize_data_file(self):
        """
        Initialize or reset the data.json file.
        """
        try:
            # Create an empty JSON structure if file doesn't exist
            if not os.path.exists(self.data_file):
                with open(self.data_file, 'w') as f:
                    json.dump([], f)
        except Exception as e:
            print(f"Error initializing data file: {e}")
            traceback.print_exc()
    
    def clean_previous_data(self):
        """
        Clean previous experiment data from data.json.
        """
        try:
            with open(self.data_file, 'w') as f:
                json.dump([], f)
            print("Previous experiment data cleared successfully.")
            # Reset epoch data
            self.epoch_data = {
                'loss_per_epoch': [],
                'accuracy_per_epoch': []
            }
        except Exception as e:
            print(f"Error cleaning previous data: {e}")
            traceback.print_exc()
    
    def log_epoch(self, loss, accuracy):
        """
        Log individual epoch data.
        
        Args:
            loss (float): Loss value for the current epoch
            accuracy (float): Accuracy value for the current epoch
        """
        self.epoch_data['loss_per_epoch'].append(loss)
        self.epoch_data['accuracy_per_epoch'].append(accuracy)
    
    def log(self, hyperparameters, metrics, model_type, model_data):
        """
        Log experiment data to data.json.
        
        Args:
            hyperparameters (dict): Hyperparameters used in the experiment
            metrics (dict): Performance metrics of the experiment
        """
        try:
            # Read existing data
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            # Prepare experiment entry
            experiment_entry = {
                'experiment_name': self.experiment_name,
                'hyperparameters': hyperparameters,
                'metrics': metrics,
                'model_type': model_type,
                'model_data': model_data,
                'epoch_data': self.epoch_data  # Include epoch-level data
            }
            
            # Add new experiment data
            data.append(experiment_entry)
            
            # Write updated data back to file
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=4)
        
        except json.JSONDecodeError:
            print("JSON Decode Error. Reinitializing data file.")
            self._initialize_data_file()
            self.log(hyperparameters, metrics)
        except Exception as e:
            print(f"Error logging data: {e}")
            traceback.print_exc()