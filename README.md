# DistLM: Distributed Training Framework for Large Language Models

## Overview

DistLM is a powerful distributed training framework designed for Large Language Models (LLMs) using Ray. It provides a scalable and efficient solution for training LLMs across multiple devices, leveraging the distributed computing capabilities of Ray.

## Features

- **Distributed Training**: Utilize multiple devices for parallel training of LLMs.
- **Ray Integration**: Leverage Ray's distributed computing framework for efficient task distribution and execution.
- **FastAPI Backend**: Robust API for device registration, task submission, and status monitoring.
- **Model and Dataset Management**: Easy upload and management of custom models and datasets.
- **React Frontend**: User-friendly interface for interacting with the training system.
- **Flexible Model Support**: Compatible with various LLM architectures, including LLaMA.

## System Architecture

DistLM consists of the following main components:

1. **Backend API** (`main.py`): FastAPI-based server handling device registration, task submission, and status queries.
2. **Ray Setup** (`ray_setup.py`): Configures and initializes the Ray cluster for distributed computing.
3. **Training Module** (`train.py`): Implements the distributed training logic using Ray.
4. **Data Loader** (`dataloader.py`): Handles loading and preprocessing of datasets.
5. **Model Loader** (`model.py`): Manages loading and initialization of LLM models.
6. **Frontend** (`App.tsx`): React-based user interface for interacting with the system.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/DistLM.git
   cd DistLM
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install Node.js and npm (for the frontend).

4. Install frontend dependencies:
   ```
   cd frontend
   npm install
   ```

## Usage

### Starting the Backend

1. Navigate to the project root directory.
2. Run the FastAPI server:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### Starting the Frontend

1. Navigate to the `frontend` directory.
2. Start the React development server:
   ```
   npm start
   ```

### API Endpoints

- `POST /devices/register`: Register a new device for distributed training.
- `GET /devices`: List all registered devices.
- `POST /tasks/submit`: Submit a new training task.
- `GET /tasks`: List all submitted tasks.
- `GET /tasks/{task_id}/status`: Check the status of a specific task.
- `POST /upload/model/`: Upload a custom model file.
- `POST /upload/dataset/`: Upload a custom dataset file.

### Distributed Training

1. Register available devices using the `/devices/register` endpoint.
2. Upload your model and dataset using the respective upload endpoints.
3. Submit a training task via the `/tasks/submit` endpoint, specifying the model, dataset, and devices to use.
4. Monitor the task status using the `/tasks/{task_id}/status` endpoint.

## Development

### Running Tests

Execute the test suite to ensure system integrity:

```
pytest test_main.py test_ray.py
```

### Adding New Models

To add support for new LLM architectures:

1. Modify `model.py` to include the new model loading logic.
2. Update `train.py` to handle the training process for the new model type.

### Extending the Frontend

The React-based frontend (`App.tsx`) can be extended to add new features or improve the user interface. Ensure to update the corresponding API calls in the frontend when modifying the backend.

## Contributing

We welcome contributions to DistLM! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your changes, ensuring to follow the existing code style.
4. Write or update tests as necessary.
5. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions, issues, or contributions, please open an issue on the GitHub repository or contact Abhi Vetukuri or Anir Vetukuri.
---

DistLM - Empowering distributed training for Large Language Models
