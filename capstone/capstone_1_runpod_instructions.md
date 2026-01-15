<!-- markdownlint-disable MD033 -->
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
1. Deploy a Pod (on-demand GPU) with an `llm-agents-from-scratch` Runpod template
2. Wait for Pod to become available and finish its setup
3. Connect to the Pod's JupyterLab service
4. Navigate to the Capstone Jupyter notebook
5. Run the notebook

### (Optional) Step 0. Create a Runpod secret for `OPENAI_API_KEY`

If you're planning to use the `OpenAILLM` for the judge LLM, you'll need to
supply an OpenAI API key. The Runpod templates that I've prepared for running
these experiments will create the necessary environment variable from a Runpod
secret.

To create the Runpod secret for your OpenAI API key, log in to the Runpod
console and click "Secrets" found in the left sidebar menu. Afterwards, click
on "Create Secret" to create a new secret.

For the Secret name, be sure to use "openai_api_key", and fill in the secret
value with your API key. Click "Create Secret" button once the necessary fields
have been populated.

<img width="1546" height="1227" alt="image" src="https://github.com/user-attachments/assets/24cf524d-9f75-4470-964a-385b29f59e83" />

### Step 1. Deploy a Pod with an `llm-agents-from-scratch` Runpod template

I've prepared two Runpod templates for Capstone 1. These templates use a Docker
image that has the necessary tools installed, including CUDA, uv, and ollama. It
also clones the book's GitHub repository and installs the
`llm-agents-from-scratch` package onto the system's Python. The two templates
differ in terms of the backbone LLM:

| Template                              | Backbone LLM     | GPU VRAM Required |
|---------------------------------------|------------------|-------------------|
| `llmagentsfromscratch-qwen3coder30b`  | Qwen3-Coder:30b  | 48GB              |
| `llmagentsfromscratch-qwen3coder480b` | Qwen3-Coder:480b | 300GB             |

To deploy a Pod with one of these templates, on the Runpod console, click on
"Pods" found in the left-hand sidebar, and then click the "+ Deploy" button.
Specify the appropriate number of GPU VRAM for the Pod that's required for the
template (i.e. backbone LLM) you'd like use.

<img width="1546" height="1227" alt="image" src="https://github.com/user-attachments/assets/55b90581-1d71-4bd2-9b86-91518f8570c6" />

After selecting your on-demand GPUs, you'll be prompted to configure the
deployment, where you can specify the Runpod template. By default, Runpod
uses a PyTorch template, so you'll need to change this to one you'd like to use.
Click on the "Change Template" button.

<img width="1546" height="1227" alt="image" src="https://github.com/user-attachments/assets/589e141a-e2fe-4ff7-afcc-6f8e3ef05f60" />

Next, you can search for the llm-agents-from-scratch template that you'd like
to use. Simply enter the name of the template in the search bar.

<img width="1546" height="1227" alt="image" src="https://github.com/user-attachments/assets/cfa91d2d-de72-4c86-ad9d-d62acbcf4add" />

The llm-agents-from-scratch template should appear. Click on it to select the
template to deploy the Pod with it. You'll be navigated back to the Configure
Deployment page.

Finally, click on the "Deploy On-Demand" button.

<img width="1546" height="1227" alt="image" src="https://github.com/user-attachments/assets/3419682b-8652-4fa0-8ad5-c6a7c855e394" />

### Step 2. Wait for Pod to become available and finish its setup

After deploying the Pod, the Docker image used by the template will be
downloaded, and its startup process will be executed. Included in this startup
is the downloading of the Ollama LLM model as well as the downloading of the
`llm-agents-from-scratch` source code and its installation into the container's
system Python. It also launches the JupyterLab service that you'll connect
with in the next step.

Go ahead and grab a cup of coffee. Expected setup times are provided in the
table below, but note that these vary depending on your selected GPU hardware.

| Template                              | Setup Time    |
|---------------------------------------|---------------|
| `llmagentsfromscratch-qwen3coder30b`  | 5–10 minutes  |
| `llmagentsfromscratch-qwen3coder480b` | 20–30 minutes |

You can monitor the setup process by viewing the Pod's system as well as
container logs. The system logs contain details on the downloading of the
Docker image, while the container logs emit updates for the startup process.

Your Pod is ready once the container logs contains details for launching the
JupyterLab service, as shown in the next screenshot.

<img width="1546" height="1227" alt="image" src="https://github.com/user-attachments/assets/5b9914d2-2d0b-43af-94f2-07405f663539" />

### Step 3. Connect to the Pod's JupyterLab service

Once startup has completed, you can connect to the JupyterLab service for your
deployed Pod. Click on the "Connect" tab, then click the JupyterLab link.

<img width="1546" height="1227" alt="image" src="https://github.com/user-attachments/assets/c258934b-9d8f-454a-b6b8-5df31c515a43" />

If successful, this should open up a new browser window exposing the typical
JupyterLab UI.

<img width="1546" height="1227" alt="image" src="https://github.com/user-attachments/assets/59c134d2-36d6-4adf-8868-34ea28a539c2" />

### Step 4. Navigate to the Capstone Jupyter notebook

As mentioned earlier, the template downloads a clone of the book's GitHub repo
into the container. All that remains now is to navigate the repo to locate and
open the Jupyter notebook for Capstone 1.

The notebook is located at: `/llm-agents-from-scratch/capstone/capstone_1_ch05.ipynb`.

<img width="1546" height="1227" alt="image" src="https://github.com/user-attachments/assets/4ebb10d4-f6bc-41e8-837c-b6c46b118303" />

### Step 5. Run the notebook

Now, instead of running this notebook on your local machine, you can run it on
your deployed Pod. Note that an Ollama service has already been launched as
part of the startup process. You should be able to click through each of the
cells and execute them successfully.

Execution times for the repeated runs section are given in the table below.

| Template                              | Execution Time (repeated trials) |
|---------------------------------------|----------------------------------|
| `llmagentsfromscratch-qwen3coder30b`  | ~15 minutes                      |
| `llmagentsfromscratch-qwen3coder480b` | ~20 minutes                      |

## IMPORTANT

Make sure to terminate your Pod when you're finished to not incur additional costs.
