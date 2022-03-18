# ENEM

# [Description]

Requirements:
* Anaconda 
* Git (terminal)

To install the environment (Windows x64) execute the following command:
```sh
conda create --name enem --file env.txt
conda activate enem
pip install -r requirements-pip.txt
```

To run the dashboard
```sh
cd dashboard
streamlit run dashboard.py
```