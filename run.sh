# export WANDB_API_KEY=16463fbb05d59374feba5b9e4b5dc705bfa4f7ba
# export WANDB_MODE=offline

hydra_args="
++model.vocoder.is_local=True
++model.vocoder.local_path=/home/tuwenming/Models/charactr/vocos-mel-24khz
++datasets.name=LibriTTS_100_360_500
++model.tokenizer=char
++ckpts.log_samples=true
++datasets.num_workers=32
"
accelerate launch --config_file 4gpu.yaml src/f5_tts/train/train_spec.py -cn F5TTS_v1_Small $hydra_args
# python -m debugpy --wait-for-client --listen 5678 `which accelerate` launch --config_file debug.yaml src/f5_tts/train/train_spec.py -cn F5TTS_v1_Small $hydra_args
# accelerate launch --config_file debug.yaml src/f5_tts/train/train_spec.py -cn F5TTS_v1_Small $hydra_args

# accelerate launch --config_file 4gpu.yaml src/f5_tts/train/train.py -cn F5TTS_Small_train $hydra_args