# Consciousness System Project

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Modules](#modules)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
The **Consciousness System Project** is an advanced AI simulation that integrates multiple machine learning techniques to mimic cognitive processes. It combines:
- **Generative Adversarial Networks (GANs)** for creative content generation
- **Deep Q-Learning (DQN)** for adaptive decision-making
- **Latent reasoning** for refining representations
- **Memory management** across multiple levels
- **Emotion simulation** for dynamic responses
- **Online data fetching** for contextual adaptation

This system is designed to process sensory inputs, generate creative outputs, and simulate conscious behavior.

---

## Features
- **Generative Creativity** – Uses GANs to produce unique embeddings from input data.  
- **Dynamic Feedback** – Implements a DQN system for real-time learning and adaptation.  
- **Latent Reasoning** – Enhances representation through GRU-based recurrent modules.  
- **Memory Management** – Supports different types of memory (sensorial, operational, long-term, etc.).  
- **Online Data Integration** – Fetches relevant data from the web using Google Custom Search API.  
- **Sandbox Simulation** – Updates an internal environment dynamically to reflect system state.  

---

## Installation
Follow these steps to set up the project:

1. Clone the repository:
```bash
git clone https://github.com/K1m1k/Scintilla.git
cd Scintilla
```

2. Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate    # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Obtain API keys for online data fetching:
- Sign up for a Google Custom Search API key.
- Replace `YOUR_GOOGLE_API_KEY` and `YOUR_CSE_ID` in the code.

5. Run the project:
```bash
python main.py
```

---

## Usage
The main entry point of the project is `main.py`. It processes a list of raw inputs and generates a conscious response.

### Example:
```python
raw_inputs = [
    "I am feeling happy today.",
    "The weather is sunny and warm.",
    "I had a great conversation with my friend.",
    "I am learning new things every day."
]

consciousness_system.simulate_consciousness(raw_inputs)
output = consciousness_system.produce_output()
print(output)
```

---

## Directory Structure
```
Scintilla/
├── gan_system/
│   ├── __init__.py
│   └── gan.py
├── dqn_system/
│   ├── __init__.py
│   └── dqn.py
├── latent_reasoning/
│   ├── __init__.py
│   └── latent_module.py
├── memory/
│   ├── __init__.py
│   └── memory.py
├── consciousness/
│   ├── __init__.py
│   └── consciousness.py
├── utils/
│   ├── __init__.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Modules
- **GAN System (`gan_system/gan.py`)** – Generates creative outputs.  
- **DQN System (`dqn_system/dqn.py`)** – Handles adaptive learning and feedback.  
- **Latent Reasoning Module (`latent_reasoning/latent_module.py`)** – Refines representations iteratively.  
- **Memory Management (`memory/memory.py`)** – Manages various memory types.  
- **Consciousness System (`consciousness/consciousness.py`)** – Integrates all components.  
- **Utilities (`utils/utils.py`)** – Provides helper functions like online data fetching.  

---

## Contributing
Contributions are welcome! Follow these steps to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them.
4. Submit a pull request.

---

## License
This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software under the terms of the MIT License. See the `LICENSE` file for details.

---

If you like this project, give it a ⭐ on GitHub!


