# Blob Game (Reinforcement Learning)
Reinforcement Learning algorithm applied on a custom terminal-based blob game.\
The Objective is to reach the target as early as possible while avoding the intended obstacles

### Setup & Usage:
1. Setup a virtual environment and install the required packages
   ```bash
   git clone https://github.com/KhalidObaide/rl-blob-ppo.git
   cd rl-blob-ppo
   python3 -m virtualenv env
   source env/bin/activate
   pip3 install requirements.txt
   ```
2. Train the model ( Shouldn't take long! ):
   ```bash
   python3 train.py
   ```
3. See the trained model in action
   ```bash
   python3 main.py
   ```

### Results: (98%)
<img width="1156" alt="Screenshot 2024-04-30 at 7 29 29 PM" src="https://github.com/KhalidObaide/rl-blob-ppo/assets/46670360/ffbc358c-b095-479e-8989-6586ca787076">

**Lives inside your terminal**:

![demo](https://github.com/KhalidObaide/rl-blob-ppo/assets/46670360/03512421-1d28-475f-957e-be54f9e97cdf)
