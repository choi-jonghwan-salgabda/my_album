mkdir -p /data/ephemeral/home/.ssh
ssh-keygen -t rsa -b 4096
ssh-copy-id -i /root/.ssh/id_rsa.pub -p 5410 owner@jongsook.iptime.org
