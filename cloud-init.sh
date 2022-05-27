#!/bin/bash

# Switch the environment.
conda activate tensorflow2_p38

# Go there.
pushd /home/ubuntu/noobrainer

# Get instance public hostname.
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` \
        && hostname=`curl -H "X-aws-ec2-metadata-token: $TOKEN" -v http://169.254.169.254/latest/meta-data/public-hostname`
echo "Here we are: ${hostname}!"

# Update streamlit About window with current ip4.
sed "s/instance_url = '.*/instance_url = '${hostname}'/" app.py > /tmp/app.py
mv /tmp/app.py .

# Start uvicorn.
nohup uvicorn --host 0.0.0.0 --port 8000 --reload main:app &

# Start streamlit.
nohup streamlit run app.py &

popd
