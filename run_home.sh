if test ! -f /home/andrew/alex_work/app_private/fairmot-serve/models/fairmot_dla34.pth; then 
fileId=1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi
fileName=/home/andrew/alex_work/app/FairMOT/models/fairmot_dla34.pth
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}
fi


if test ! -f /home/andrew/alex_work/app_private/fairmot-serve/models/crowdhuman_dla34.pth; then 
fileId=1SFOhg_vos_xSYHLMTDGFVZBYjo8cr2fG
fileName=/home/andrew/alex_work/app/FairMOT/models/crowdhuman_dla34.pth
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}
fi



docker run \
    --rm \
    -ti \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
    --shm-size='1G' \
    -e PYTHONUNBUFFERED='1' \
    -e SERVER_ADDRESS='http://78.46.75.100:38585' \
    -e API_TOKEN='0e3NJFlXteWPp1PAtA5fDrIkdtGeh42iTSVd6J2ecMn1hZW497M1SWd2DLEHhdUDaliPVe6eI9SChBZLBsKTVyTlgrsn240eJ9gObtUMis2vmSsFDr5yhssFX3iHtStJ' \
    -v /home/andrew/alex_work/app/FairMOT:/alex_work \
    -v /usr/local/cuda-9.1:/usr/local/cuda \
    -v /opt/pycharm:/pycharm \
    -v /home/andrew/pycharm-settings/smtool_gui:/root/.PyCharmCE2018.2 \
    -v /home/andrew/pycharm-settings/smtool_gui__idea:/workdir/.idea \
    fairmot
