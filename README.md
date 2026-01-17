# EPL-thesis
Multi-agent system for the thesis.

`author:` Stefanos Panteli
`github:` https://github.com/stefanosPanteli/EPL-thesis

# Run Steps
## To use:
1. First make a virtual envirnment.
2. Then install the requirements.
3. Make the API keys, like the `.env.example` file.
    Where to find the keys:
    1. Open Router:
        - To create an account, sign up [https://openrouter.ai/](https://openrouter.ai/).
        - To create an API key, go to [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys).
    2. Tavily:
        - To create an account, sign up [https://tavily.com/](https://tavily.com/).
        - To create an API key, go to [https://app.tavily.com/home](https://app.tavily.com/home).
    3. LangSmith/Chain:
        - To create an account, sign up [https://smith.langchain.com/](https://smith.langchain.com/).
        - To create an API key, log in -> Settings -> API Keys
```bash
cd .\to the clone directory

python -m venv venv # Creates the virtual environment named venv
.\venv\Scripts\Activate.ps1 # Activate the venv
pip install -r requirements.txt # Installs the requirements

#! Modify the .env.example file
cp .env.example .env # Copy the .env.example file to .env
```

## To run:
1. Run the following commands each time you want to start the system.
(From the Clone directory)
```bash
cd .\to the clone directory
.\venv\Scripts\Activate.ps1 # Activate the venv

$env:PYTHONPATH = (Get-Location) # Set the PYTHONPATH (For windows)
#! OR
export PYTHONPATH=$PWD           # Set the PYTHONPATH (For linux)


cd /agents/<agent_name> # Go to the agent folder
python <agent_name>.py # Run the agent
```

# Agents # TODO: Add a README.md to each agent
# Workflow

## TODO
- Add readme.md to each agent
- Add a workflow diagram to each agent and here
- add an overview here

- test code_annotator on a more complex workflow
- make a qa
- make the main workflow

- add a memory to all llms