# llm-agents-from-scratch Runpod Template

## Creating Runpod Template

## Runpod setup

1. Create a **Custom Template** pod with `llmagentsfromscratch/runpod:latest`.
2. Ensure a **persistent volume** is mounted at `/workspace` (default in Runpod).
3. Set env vars as needed: `OLLAMA_MODEL`, `JUPYTER_PASSWORD` (used for
spinning up Jupyter notebooks through runpod UI).
4. Set TCP port to 22
5. Add credentials for pulling from `llmagentsfromscratch/runpod:latest` i.e,. get a Docker PAT
6. SSH in using the endpoint/port Runpod provides.

```sh
# build docker
docker build -t llmagentsfromscratch/runpod:latest -f docker/runpod/Dockerfile .

# run docker
docker run -e DEV=1 \
  -e OLLAMA_MODEL=qwen3:8b \
  -e JUPYTER_PASSWORD=llmagentsfromscratch \
  -p 8888:8888 \
  -it llmagentsfromscratch/runpod:latest
```

## References

- <https://docs.runpod.io/pods/configuration/use-ssh#full-ssh-via-public-ip-with-key-authentication>
