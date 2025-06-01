df -k
sudo fdisk -l
mkdir SambaData/Home
mkdir -p SambaData/Home
sudo mount /dev/sdb1 ~/SambaData/Home/
mkdir -p SambaData/Data1
sudo mount /dev/sdb2 ~/SambaData/Data1
