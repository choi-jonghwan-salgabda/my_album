import paramiko
import os
import stat # 파일 속성 확인을 위해 추가

def download_all_files_from_dir(hostname, port, username, password, remote_directory, local_directory):
    """
    SFTP 서버의 특정 디렉토리에 있는 모든 파일을 지정된 로컬 디렉토리로 다운로드합니다.

    Args:
        hostname (str): SFTP 서버 호스트 이름 또는 IP 주소.
        port (int): SFTP 포트 번호.
        username (str): SFTP 사용자 이름.
        password (str): SFTP 비밀번호.
        remote_directory (str): 파일을 다운로드할 원격 디렉토리 경로.
        local_directory (str): 파일을 저장할 로컬 디렉토리 경로.
    """
    transport = None
    sftp = None
    downloaded_count = 0
    skipped_count = 0

    try:
        # --- SFTP 연결 ---
        print(f"Connecting to {hostname}:{port}...")
        transport = paramiko.Transport((hostname, port))
        # 비밀번호 대신 SSH 키 사용을 강력히 권장합니다.
        # 예: transport.connect(username=username, pkey=paramiko.RSAKey.from_private_key_file('/path/to/your/private_key'))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        print("Connection successful.")

        # --- 로컬 디렉토리 생성 ---
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)
            print(f"Created local directory: {local_directory}")
        else:
            print(f"Local directory already exists: {local_directory}")

        # --- 원격 디렉토리 파일 목록 가져오기 ---
        print(f"Listing files in remote directory: {remote_directory}...")
        try:
            # listdir_attr를 사용하면 파일/디렉토리 구분 및 속성 확인 가능
            remote_items = sftp.listdir_attr(remote_directory)
            print(f"Found {len(remote_items)} items in remote directory.")
        except IOError as e:
            print(f"❌ Error listing remote directory '{remote_directory}': {e}")
            print("Please check if the remote directory path is correct and you have permissions.")
            return # 디렉토리 접근 불가 시 함수 종료

        # --- 파일 다운로드 ---
        for item_attr in remote_items:
            # 디렉토리는 건너뛰고 파일만 처리
            if not stat.S_ISREG(item_attr.st_mode):
                print(f"  Skipping (not a file): {item_attr.filename}")
                skipped_count += 1
                continue

            remote_file_path = f"{remote_directory}/{item_attr.filename}" # 원격 경로 조합 (SFTP는 보통 / 사용)
            local_file_path = os.path.join(local_directory, item_attr.filename)

            try:
                print(f"  Downloading {remote_file_path} to {local_file_path}...")
                sftp.get(remote_file_path, local_file_path)
                downloaded_count += 1
            except Exception as e:
                print(f"  ⚠️ Error downloading {item_attr.filename}: {e}")
                skipped_count += 1

        print("\n--- Download Summary ---")
        print(f"Successfully downloaded: {downloaded_count} files")
        print(f"Skipped or failed: {skipped_count} items")
        print("------------------------")

    except paramiko.AuthenticationException:
        print("❌ Authentication failed. Please check your username and password.")
    except paramiko.SSHException as sshException:
        print(f"❌ Could not establish SSH connection: {sshException}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        # --- 연결 종료 ---
        if sftp:
            sftp.close()
            print("SFTP connection closed.")
        if transport and transport.is_active():
            transport.close()
            print("Transport closed.")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 사용 예시
    # 설정 값 기본적인 확인
    download_all_files_from_dir(
    hostname = '192.168.219.10',   # SFTP 서버 주소
    port = 5410,   # SFTP 포트 (기본값: 22)
    username = 'ftpuser',   # SFTP 사용자 이름
    password = '029095277',   # SFTP  
    remote_directory = './SambaData/OwnerData/내가족/',           # 로컬 파일 목록 경로
    local_directory = '/data/ephemeral/home/data/images/train/'     # 파일을 저장할 로컬 디렉토리
    )