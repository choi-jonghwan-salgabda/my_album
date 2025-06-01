#!/bin/bash


export`KEY_COUNTRY="KR"
export KEY_PROVINCE="Seoul"
export KEY_CITY="Dondeamun-Gu"
export KEY_ORG="MyHomeCompeny"
export KEY_EMAIL="salgabda@naver.com"
export KEY_OU="MyFamily"


./easyrsa init-pki
./easyrsa build-ca nopass
./easyrsa gen-req server nopass
./easyrsa sign-req server server
./easyrsa gen-dh
