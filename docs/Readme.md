# Group 10 Project 2.1

# Switched from Docker to Conda (Plan is outdated)
1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Clone the repo:
   git clone https://github.com/VandingenenLars/ml-agents-project-2.1.git
3. Create and activate the environment:
   conda env create -f environment.yml
   conda activate ml-agents
4. Run training:
   mlagents-learn config/ppo.yaml --run-id=MyRun
5. Open Unity and press Play.
