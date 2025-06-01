
#!/bin/bash


# 설정 값
VPN_PORT=6000
VPN_PROTO=udp
VPN_SERVER_NAME="vpn_my_home"
CLIENT_NAME="ClientMyHome"

	echo "# OpenVPN 및 Easy-RSA 설치"

sudo apt update
sudo apt install -y openvpn easy-rsa

	echo "# Easy-RSA 디렉토리 생성 및 초기화"

make-cadir ~/openvpn-ca
cd ~/openvpn-ca || exit



echo "# Easy-RSA 설정 파일 수정"

#bash -c cat > vars << EOF
#set_var KEY_COUNTRY "KR"
#set_var KEY_PROVINCE "Seoul"
#set_var KEY_CITY "Dondeamun-Gu"
#set_var KEY_ORG "MyHomeCompeny"
#set_var KEY_EMAIL "salgabda@naver.com"
#set_var KEY_OU "MyFamily"
#EOF


	echo "# CA(인증기관) 생성 (자동 확인)"
./easyrsa init-pki

./easyrsa --batch build-ca nopass

	echo "# 서버 키 및 인증서 생성 (자동 입력)"

./easyrsa --batch gen-req "$VPN_SERVER_NAME" nopass
./easyrsa --batch sign-req server "$VPN_SERVER_NAME"

	echo "# 클라이언트 키 및 인증서 생성 (자동 입력)"

./easyrsa --batch gen-req "$CLIENT_NAME" nopass
./easyrsa --batch sign-req client "$CLIENT_NAME"

	echo "# Diffie-Hellman 키 및 TLS-Auth 키 생성"

./easyrsa gen-dh
openvpn --genkey --secret ta.key

	echo "# PEM 파일 생성 (클라이언트용 단일 파일)"
cat pki/ca.crt pki/issued/"$CLIENT_NAME".crt pki/private/"$CLIENT_NAME".key ta.key > ~/openvpn-ca/"$CLIENT_NAME".pem

	echo "# OpenVPN 서버 구성"

sudo cp /usr/share/doc/openvpn/examples/sample-config-files/server.conf.gz /etc/openvpn/
sudo gzip -d /etc/openvpn/server.conf.gz
sudo bash -c  cat > /etc/openvpn/server.conf <<EOF
port $VPN_PORT
proto $VPN_PROTO
dev tun
ca /etc/openvpn/ca.crt
cert /etc/openvpn/$VPN_SERVER_NAME.crt
key /etc/openvpn/$VPN_SERVER_NAME.key
dh /etc/openvpn/dh.pem
tls-auth /etc/openvpn/ta.key 0
keepalive 10 120
persist-key
persist-tun
cipher AES-256-CBC
user nobody
group nogroup
EOF

	echo "# 인증서 및 키 파일 복사"

sudo cp pki/ca.crt pki/issued/"$VPN_SERVER_NAME".crt pki/private/"$VPN_SERVER_NAME".key pki/dh.pem ta.key /etc/openvpn/


#	echo("# IP 포워딩 활성화"

#sudo sed -i '/net.ipv4.ip_forward/s/^#//' /etc/sysctl.conf
#sudo sysctl -p

#	echo "# 방화벽 규칙 추가"

#sudo ufw allow "$VPN_PORT/$VPN_PROTO"
#sudo ufw enable


	echo "# OpenVPN 서비스 시작"

sudo systemctl start openvpn@server
sudo systemctl enable openvpn@server

	echo "클라이언트 인증서 파일 생성 완료: ~/openvpn-ca/$CLIENT_NAME.pem"
