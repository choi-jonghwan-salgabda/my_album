import subprocess

# fc-list 명령어를 사용하여 한글이 지원되는 폰트를 리스트로 가져오기
result = subprocess.run(["fc-list", ":lang=ko", "--format=%{file}\n"], stdout=subprocess.PIPE, text=True)
font_list = result.stdout.splitlines()
print("설치된 한글 폰트 목록:")
for font in font_list:
    print(font)