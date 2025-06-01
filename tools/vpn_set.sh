 sudo apt update 
 sudo apt install openvpn easy-rsa
 make-cadir ~/openvpn-ca
 cd ~/openvpn-ca/

 ls -al
 vim vars
 ./easyrsa init-pki
 ./easyrsa build-ca nopass
 ./easyrsa gen-req server nopass
 ./easyrsa sign-req server server
 ./easyrsa gen-req client1 nopass
 ./easyrsa sign-req client client1
 ./easyrsa gen-dh
 openvpn --genkey --secret ta.key
 cat pki/ca.crt pki/issued/client1.crt pki/private/client1.key ta.key > client1.pem
 sudo cp /usr/share/doc/openvpn/examples/sample-config-files/server.conf.gz /etc/openvpn/
 sudo gzip -d /etc/openvpn/server.conf.gz
 sudo vim /etc/openvpn/server.conf
 sudo cp pki/ca.crt pki/issued/server.crt pki/private/server.key pki/dh.pem ta.key /etc/openvpn/
 sudo systemctl start openvpn@server
 sudo systemctl enable openvpn@server
 sudo ufw allow 6000/tcp

