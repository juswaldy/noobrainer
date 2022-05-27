# noobrainer

This here is a repo for our MLE capstone project at [fourthbrain.ai](https://www.fourthbrain.ai/), cohort 6. The domain we have chosen is Natural Language Processing, specifically Automated Metadata Tagging and Topic Modeling.


## Usage

### On your local machine

### 1. Minimum requirements

- 16G of RAM
- [CUDA enabled GPU](https://developer.nvidia.com/cuda-gpus) with at least 4GB of memory
- Ubuntu 18.04 LTS
- Python 3.8

### 2. Clone this repo

```bash
git clone https://github.com/juswaldy/noobrainer.git
cd noobrainer
```

### 3. Install requirements in a conda environment

```bash
conda create -n noobrainer python=3.8
source activate noobrainer
pip install --no-cache-dir -r requirements.txt
```

### 4. Configure AWS command line interface and download models and data from s3

```bash
aws configure
aws s3 cp s3://noobrainer/models/ner-healthtechother-titles-23.pkl ./models/ner-healthtechother-titles-23.pkl
aws s3 cp s3://noobrainer/models/tomo-60k.pkl ./models/tomo-60k.pkl
aws s3 cp s3://noobrainer/models/tomo-all-87k-articles-single-21.pkl ./models/tomo-healthtech-articles-single-17.pkl
aws s3 cp s3://noobrainer/models/tomo-healthtech-titles-single-17.pkl ./models/tomo-healthtech-titles-single-17.pkl
aws s3 cp s3://noobrainer/models/tomo-healthtech-articles-single-17.pkl ./models/tomo-healthtech-articles-single-17.pkl
aws s3 cp s3://noobrainer/data/health_tech_time.csv ./data/health_tech_time.csv
aws s3 cp s3://noobrainer/data/0_combined_set_60k_date.csv ./data/0_combined_set_60k_date.csv
```

### 5. Start up the API server and the frontend

```bash
# Uvicorn on default port 8000.
nohup uvicorn --host 0.0.0.0 --port 8000 --reload main:app &

# Streamlit on default port 8501.
nohup streamlit run app.py &
```

The API server is running on port `8000`. The frontend is running on port `8501`.


### On AWS EC2

This is how we deployed our first demo version and published it.

<details><summary>Show/hide</summary>

### 1. Create an EC2 instance and ssh into it

- Choose image type: `Deep Learning AMI (Ubuntu 18.04) Version 60.2`
- Choose instance type: `g4dn.*`. The one we use for the demo is `g4dn.2xlarge`, but any size >= `large` should work fine. Note: We tried using `g4ad` instances and had trouble getting it to work with the GPU
- Copy its public ip dns to clipboard
- ssh into it

### 2. Clone this repo

```bash
git clone https://github.com/juswaldy/noobrainer.git
cd noobrainer
```

### 3. Activate GPU environment and install requirements

```bash
source activate tensorflow2_p38
pip install --no-cache-dir -r requirements.txt
```

### Continue with Step 4 and 5 above

Don't forget to stop the instance when you're not using it anymore.

</details>


## Shoutouts

- [Fourthbrain.ai](https://www.fourthbrain.ai/) staff especially Bruno Janota!
- LDA Topic Models [video](https://www.youtube.com/watch?v=3mHy4OSyRf0) by Andrius Knispelis!
- [Top2Vec](https://github.com/ddangelov/Top2Vec) by Dimo Angelov!
- [fastapi](https://fastapi.tiangolo.com/) by Sebastián Ramírez!
- [streamlit](https://streamlit.io/) by Streamlit!
- [python-stat-tools](https://github.com/harmkenn/python-stat-tools) by Ken Harmon!
- [st-click-detector](https://github.com/vivien000/st-click-detector) by Vivien Tran-Thien!
- University of South Florida [ClipArt ETC](https://etc.usf.edu/clipart/)!
