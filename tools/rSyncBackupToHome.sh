echo  ~HOME
rsync -az  -e 'ssh -p 5410' 			   owner@rnaxj.iptime.org:./				SambaData/OwnerData
echo  '공유자료'
rsync -az  -e 'ssh -p 5410' 			   owner@rnaxj.iptime.org:/home/Data00/Data/ 		SambaData/ShareData
echo  '재미이반'
rsync -az  -e 'ssh -p 5410' --exclude '지난자료들' owner@rnaxj.iptime.org:/home/Data10/Data/ 		SambaData/SsamData/Data0
rsync -az  -e 'ssh -p 5410' 			   owner@rnaxj.iptime.org:/home/Data10/Data/지난자료들  SambaData/Backup/Document/재미난
echo  '재롱이사랑해세반'
rsync -az  -e 'ssh -p 5410' --exclude '지난자료들' owner@rnaxj.iptime.org:/home/Data11/Data/ 		SambaData/SsamData/Data1
rsync -az  -e 'ssh -p 5410' 			   owner@rnaxj.iptime.org:/home/Data11/Data/지난자료들  SambaData/Backup/Document/사랑해
echo  '행복한반'
rsync -az  -e 'ssh -p 5410' --exclude '지난자료들' owner@rnaxj.iptime.org:/home/Data20/Data/ 		SambaData/SsamData/Data2
rsync -az  -e 'ssh -p 5410' 			   owner@rnaxj.iptime.org:/home/Data20/Data/지난자료들  SambaData/Backup/Document/행복한
echo  '즐거운반'
rsync -az  -e 'ssh -p 5410' --exclude '지난자료들' owner@rnaxj.iptime.org:/home/Data21/Data/ 		SambaData/SsamData/Data21
rsync -az  -e 'ssh -p 5410' 			   owner@rnaxj.iptime.org:/home/Data21/Data/지난자료들  SambaData/Backup/Document/즐거운
echo  '신나는반'
rsync -az  -e 'ssh -p 5410' --exclude '지난자료들' owner@rnaxj.iptime.org:/home/Data30/Data/ 		SambaData/SsamData/Data3
rsync -az  -e 'ssh -p 5410' 			   owner@rnaxj.iptime.org:/home/Data30/Data/지난자료들  SambaData/Backup/Document/신나는
