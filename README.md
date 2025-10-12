# American Sign Language (ASL) Recognition

This project is a web-based American Sign Language recognition system built with Flask for the backend and PyTorch for deep learning. Users can interact with a trained model through a web interface to recognize ASL gestures.

---

## Project Structure

- app.py : Flask backend
- index.html : Frontend UI
- helper_functions.py : Utility functions for the model
- models/ : Folder to store trained models
- final_model/ : Your final trained model
- asl_alphabet_data/ : Dataset folder
- requirements.txt : Python dependencies
- venv/ : Python virtual environment (not tracked)

---

## Prerequisites

- Python 3.11+
- Git (optional, for cloning the repo)
- Virtual environment tool (`venv`)

---

## Installation and Setup

1. Clone the repository:

$ git clone https://github.com/jasonflores04/ASL_Recognition.git
$ cd ASL_Recognition

3. Create a virtual environment:

$ python3 -m venv venv

3. Activate the virtual environment:

- macOS/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

4. Install dependencies:

$ pip install -r requirements.txt


---

## Running the Application

1. Make sure the virtual environment is activated.
2. Start the Flask server:
python3 app.py

4. Open your browser and go to: `http://127.0.0.1:5000` to see the frontend and interact with the ASL model.

---

## Dataset

The ASL dataset should be placed in `asl_alphabet_data/asl_alphabet_train/asl_alphabet_train`.  
You can download it from Kaggle: [ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet)

---

## Trained Models

The `final_model/` folder contains the trained model.  
Modify `app.py` or your notebook if you want to retrain.

---

## Notes & Troubleshooting

- Keep the folder structure intact for Flask to find frontend and model files.  
- If you see `ModuleNotFoundError`, ensure all packages are installed from `requirements.txt`.  
- On Mac, you may need `brew install cmake` for some PyTorch dependencies.

---

## References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Documentation](https://pytorch.org/)
- [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet)

---

## Contributing

Fork the repository and make improvements. Update `requirements.txt` if adding new packages.
