from pydantic import BaseModel
from typing import List, Dict
import os, json

class ProcessingInput(BaseModel):
    config_path_train: str
    config_path_valid: str

class ProcessingOutput(BaseModel):
    out_dir: str    

class TrainingParameters(BaseModel):
    batch_size: int
    epochs: int
    steps_per_epoch: int
    steps_per_valid: int
    num_sub_epoch: int
    sample_size: int
    loss: str
    class_weights: Dict[str, int]

class LearningRateParameters(BaseModel):
    learning_rate: float    
    min_learning_rate: float
    monitor: str
    factor: float
    patience: int

class ModelParameters(BaseModel):
    architecture: str
    layers: List[int]
    droput_rate: float
    activation_function: str

class ExperimentConfig(BaseModel):
    input: ProcessingInput
    output: ProcessingOutput
    model_params: ModelParameters
    training_params: TrainingParameters
    learning_params: LearningRateParameters

    def save_to_json(self):
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Define the path to the JSON file
        json_file_path = os.path.join(self.output_dir, "experiment_config.json")

        # Save the combined parameters to a JSON file
        with open(json_file_path, 'w') as f:
            json.dump(self.dict(), f, indent=4)

        print(f"Configuration saved to {json_file_path}")    