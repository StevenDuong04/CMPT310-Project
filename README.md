# CMPT310-Project

## Initialization
How to set up virtual environment and install libraries.

Have a copy of this repository in your computer and then open it in VSCode
1. Installing python virtual environment
MacOS
```bash
python3 -m venv venv
```

Windows
```bash
python -m venv venv
```

2. Activate virtual environment (Before you start working)
MacOS
```bash
source venv/bin/activate
```

Windows (Powershell)
```bash
venv\Scripts\Activate.ps1
```

Windows (Command Prompt)
```bash
venv\Scripts\activate
```

3. Install python libraries (Should have the requirements.txt in your folder)
```bash
pip install -r requirements.txt
```

4. When you are done with your session
```bash
deactivate
```

## Running Streamlit App
1. To run a streamlit application into a browser
```bash
streamlit run app_name.py
```

1. To exit the application when done
Use keys on your keyboard: ctrl + c
