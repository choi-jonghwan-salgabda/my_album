#!/bin/bash

IF_DENUG=True

echo "이 스크립트는 Windows에서 사용하기 위한 .ovpn 파일을 PEM 파일 방식으로
간단히 생성하도록 작성되었습니다."

echo "# 설정 값"

VPN_SERVER_IP="192.168.219.10"
VPN_PORT=6000
VPN_PROTO=udp
CLIENT_NAME="ClientMyHome"
PEM_FILE_PATH="./ClientMyHome.pem"

echo "# OpenVPN 클라이언트 구성 파일 생성"

cd ~/openvpn-ca || exit

CONFIG_FILE="./$CLIENT_NAME.ovpn"
cat > "$CONFIG_FILE" <<EOF
client
dev tun
proto $VPN_PROTO
remote $VPN_SERVER_IP $VPN_PORT
resolv-retry infinite
nobind
persist-key
persist-tun
cipher AES-256-CBC
key-direction 1
<cert>
$(cat "$PEM_FILE_PATH")
</cert>
verb 3
EOF

echo "# 결과 출력"

echo "클라이언트 OVPN 구성 파일 생성 완료: $CONFIG_FILE"
echo "이 파일을 Windows OpenVPN GUI 설정 디렉토리에 복사하세요."

