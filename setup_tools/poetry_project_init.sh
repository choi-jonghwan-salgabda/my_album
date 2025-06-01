 mkdir shared_utils
 cd shared_utils
 poetry init # 필요한 정보 입력
 ls
 mkdir sorc
 cp ../my_album/utility ./sorc
 cp -r ../my_album/utility ./sorc
 cd sorc/
 ls
 cd utility/
 ls
 cd ../my_yolo_tiny/
 poetry add ../shared_utils # shared_utils 프로젝트 디렉토리의 상대 경로 지정
 cd ../shared_utils/sorc/utility/
