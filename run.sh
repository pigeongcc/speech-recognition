eval "$(conda shell.bash hook)"
conda activate asr_project_min

cd /root/speech-recognition/deploy

flask run --host=185.188.183.103 &

disown