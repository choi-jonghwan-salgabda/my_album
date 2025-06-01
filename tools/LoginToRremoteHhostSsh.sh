  ssh-keygen -t rsa 
  cp ~/.ssh/id_rsa ~/.ssh/id_rsa.enc
  openssl rsa -in ~/.ssh/id_rsa.enc -out ~/.ssh/id_rsa
  ssh-copy-id -i ~/.ssh/id_rsa.pub  -p 5410 owner@rnaxj.iptime.org
