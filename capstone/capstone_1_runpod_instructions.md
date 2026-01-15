# Running Capstone 1 Experiments on Runpod

This document contains instructions for running Capstone 1 experiments using
Runpod on-demand GPUs. You'll spin up a Runpod pod with the correct hardware
requirements and execute this Capstone's Jupyter notebook on it.

## What you'll need

1. A Runpod account with credits ($20 of credits should be enough to run
Qwen3-Coder:480B)
2. An OpenAI API key (Optional, recommended for the judge LLM)

## High-level steps

0. (Optional) Create a Runpod secret for OPENAI_API_KEY
1. Deploy a Pod (on-demand GPU) with one of the `llm-agents-from-scratch`
templates
2. Wait for Pod to become available and finish its setup
3. Connect to the Pod's JupyterLab service
4. Navigate to the Capstone Jupyter notebook
5. Run the notebook

### (Optional) Step 0. Create a Runpod secret for `OPENAI_API_KEY`

If you're planning to use the `OpenAILLM` for the judge LLM, you'll need to
supply an OpenAI API key. The Runpod templates that I've prepared for running
these experiments will create the necessary environment variable from a Runpod
secret.

To create the Runpod secret for your OpenAI API key, login into the Runpod
console and click "Secrets" found in the left side-bar menu. Afterwards, click
on "Create Secret" to create a new secret.

For the Secret name, be sure to use "openai_api_key", and fill in the secret
value with your API key.

<img width="1546" height="1227" alt="image" src="https://github.com/user-attachments/assets/24cf524d-9f75-4470-964a-385b29f59e83" />
