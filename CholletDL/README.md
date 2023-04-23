# Code written while reading chollets book

## Run via devcontainer:

1. Run "docker build -t cholletdev ."
2. Create .devcontainer/devcontainer.json:
```
{
  "image": "cholletdev",
  "runArgs": ["--gpus=all"]
}
```
3. Open vscode here, with the docker devcontainer extension installed.