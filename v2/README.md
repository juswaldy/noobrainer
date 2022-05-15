# v2

## Usage:

### 1. Clone this repo

```bash
# Go to your special folder
cd /tmp

# Clone this repo
git clone https://github.com/juswaldy/noobrainer.git

# Go into v2 folder
cd noobrainer/v2
```

### 2. Build docker image

```bash
docker build -t noobrainer:latest .
```

### 3. Create a `models` folder and download the pretrained models into it

```bash
# Create a models folder and go there
mkdir models
cd models

# Install gdown
pip install gdown --upgrade

# Download the pretrained models
gdown https://drive.google.com/uc?id=1P1iYqQ_y3SzZcAVMsh4lsjKicgbIfuWJ
```

### 4. Stand up the API

```bash
# Run the docker image, remember your special folder
docker run -it --gpus all --rm -v /tmp/noobrainer/v2/models:/models -p 8000:8000 noobrainer:latest
```

### 5. Visit the API docs

Point your browser to http://localhost:8000/docs.

![docs](https://raw.githubusercontent.com/juswaldy/noobrainer/v2/docs.PNG)
