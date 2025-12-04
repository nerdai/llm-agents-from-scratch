# llm-agents-from-scratch Runpod Template

```sh
# build docker
docker build -t llmagentsfromscratch/runpod:latest -f docker/runpod/Dockerfile .

# run docker
docker run -e DEV=1 \
  -e OLLAMA_MODELS=qwen3:8b \
  -e JUPYTER_PASSWORD=llmagentsfromscratch \
  -p 8888:8888 \
  -it llmagentsfromscratch/runpod:latest
```

For multiple models: `OLLAMA_MODELS=qwen2.5:7b,llama3.1:8b`

## References

- <https://docs.runpod.io/pods/configuration/use-ssh#full-ssh-via-public-ip-with-key-authentication>
