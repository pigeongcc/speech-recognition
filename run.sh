eval "$(conda shell.bash hook)"
conda activate asr_project_min

cd /root/speech-recognition/deploy

flask_pid=$(pstree -p | grep -P -o 'flask\([0-9]+\)' | grep -o '[0-9]\+')
kill -s 9 $flask_pid

nohup flask run --host=185.188.183.103 &