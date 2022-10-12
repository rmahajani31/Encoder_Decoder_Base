sudo apt-get -y install git
sudo apt install unzip
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
sh Anaconda3-2021.05-Linux-x86_64.sh
conda env create -f Encoder_Decoder_Base/encoder_decoder_base_env.yml
conda activate encoder_decoder_base_env
python -m spacy download en_core_web_lg
rm ~/Anaconda3-2021.05-Linux-x86_64.sh
