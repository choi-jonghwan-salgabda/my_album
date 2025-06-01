import paramiko
import os

def download_files_sftp(hostname, port, username, password, remote_files, local_directory):
    # SFTP 클라이언트 설정
    try:
        # SSH 클라이언트 생성
        transport = paramiko.Transport((hostname, port))
        transport.connect(username=username, password=password)

        # SFTP 세션 시작
        sftp = paramiko.SFTPClient.from_transport(transport)

        # 로컬 디렉토리 생성 (존재하지 않는 경우)
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)

        # 파일 다운로드
        for remote_file in remote_files:
            local_file_path = os.path.join(local_directory, os.path.basename(remote_file))
            print(f"Downloading {remote_file} to {local_file_path}...")
            sftp.get(remote_file, local_file_path)

        print("모든 파일 다운로드 완료.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # SFTP 세션 종료
        sftp.close()
        transport.close()

def read_file_list(file_path):
    # 로컬 파일 목록 읽기
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def read_file_list_execpt_ext(file_path):
    # 로컬 파일 목록 읽기
    with open(file_path, 'r') as file:
        return [os.path.splitext(line.strip())[0] for line in file if line.strip()]

# 사용 예시
hostname = '192.168.219.10'  # SFTP 서버 주소
port = 5410  # SFTP 포트 (기본값: 22)
username = 'ftpuser'  # SFTP 사용자 이름
password = '029095277'  # SFTP  
file_list_path = '/data/ephemeral/home/data/json_file_list/'    # 로컬 파일 목록 경로
local_directory = '/data/ephemeral/home/data/train/image'       # 파일을 저장할 로컬 디렉토리

# 로컬 파일 목록 읽기
remote_files = read_file_list_execpt_ext(file_list_path)

# 파일 다운로드
download_files_sftp(hostname, port, username, password, remote_files, local_directory)


# remote_directory  = ['home/owner/SambaData/Backup/FastCamp/aihub_data/datafiles/1.traing'  # 다운로드할 원격 파일 목록
# local_directory = '/data/ephemeral/home.data/train'  # 파일을 저장할 로컬 디렉토리
# ~/SambaData/Backup/FastCamp/aihub_data/datafiles
# # SFTP 연결 및 파일 목록 가져오기
# try:
#     transport = paramiko.Transport((hostname, port))
#     transport.connect(username=username, password=password)
#     sftp = paramiko.SFTPClient.from_transport(transport)

#     # 원격 디렉토리의 파일 목록 가져오기
#     remote_files = [os.path.join(remote_directory, f) for f in get_remote_file_list(sftp, remote_directory)]

#     # 파일 다운로드
#     download_files_sftp(hostname, port, username, password, remote_files, local_directory)
# finally:
#     # SFTP 세션 종료
#     sftp.close()
#     transport.close()

