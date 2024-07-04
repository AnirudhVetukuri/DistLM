import ray
from ray_module.train import train_model

def test_ray_training():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Submit a training task with default model and dataset
    result_ref = train_model.remote(epochs=1)

    # Wait for the task to complete
    result = ray.get(result_ref)

    # Check the result
    assert result == "Training complete"

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    test_ray_training()
