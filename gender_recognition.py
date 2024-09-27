import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
import srt
import io
import numpy as np
from flask import Flask, request, jsonify
from datetime import timedelta
from src.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification
from pydub import AudioSegment

app = Flask(__name__)

# 配置和模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "m3hrdadfi/hubert-base-persian-speech-gender-recognition"
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
model = HubertForSpeechClassification.from_pretrained(model_name_or_path).to(device)

# 定义方法：将音频文件加载为数组并重新采样
def speech_file_to_array_fn(wav_bytes, sampling_rate):
    # print("222")
    # 将字节流转换为tensor
    speech_array, _sampling_rate = torchaudio.load(io.BytesIO(wav_bytes))
    # print("sampling_rate=", sampling_rate)
    # print("_sampling_rate=", _sampling_rate)
    # resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    resampler = torchaudio.transforms.Resample(sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

# 定义预测方法
def predict_gender(speech, sampling_rate):
    # print("555")
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}
    # print("666")
    with torch.no_grad():
        logits = model(**inputs).logits

    # print("777")
    # 计算每个类别的概率
    print(logits)
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Label": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    print(outputs)
    # 根据分数选择性别
    gender = 'M' if np.argmax(scores) == 1 else 'F'
    return gender

# 定义时间格式转换
def format_timestamp(time_delta):
    total_seconds = int(time_delta.total_seconds())
    return str(timedelta(seconds=total_seconds))

# 分割WAV文件的函数
def slice_wav_by_timestamp(wav_audio, start_ms, end_ms, start_timestamp):
    audio_segment = wav_audio[start_ms:end_ms]

    # 创建一个 BytesIO 对象，用于保存字节流
    audio_bytes_io = io.BytesIO()

    # 将 AudioSegment 导出为字节流，格式可以是 wav 或其他支持的格式
    audio_segment.export(audio_bytes_io, format="wav")


    # 保存为 .wav 文件
    # audio_segment.export(f"temp/a_{start_timestamp}.wav", format="wav")
    audio_segment.export(f"temp.wav", format="wav")

    # 获取字节数据
    audio_bytes = audio_bytes_io.getvalue()
    return audio_bytes

def speech_file_to_array_fn1(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


# Flask API方法
@app.route('/speaker_recognition', methods=['POST'])
def speaker_recognition():
    wav_file = request.files['wav']
    srt_file = request.files['srt']

    # 读取srt文件内容
    srt_content = srt_file.read().decode('utf-8')
    subtitles = list(srt.parse(srt_content))

    # 读取wav音频数据
    # wav_bytes = wav_file.read()
    audio = AudioSegment.from_wav(io.BytesIO(wav_file.read()))

    sampling_rate = feature_extractor.sampling_rate

    response_data = {}

    # print("111")
    # 处理每个字幕时间段
    for subtitle in subtitles:
        print(subtitle)
        try:            
            start_time_ms = int(subtitle.start.total_seconds() * 1000)
            end_time_ms = int(subtitle.end.total_seconds() * 1000)

            # 格式化时间戳为HH:mm:ss
            start_timestamp = format_timestamp(subtitle.start)
            end_timestamp = format_timestamp(subtitle.end)

            # 根据时间段分割音频
            print(f'start_time_ms={start_time_ms}, end_time_ms={end_time_ms}')
            audio_segment = slice_wav_by_timestamp(audio, start_time_ms, end_time_ms, start_timestamp)

            # 从WAV文件分割相应的音频片段
            # audio_segment = speech_file_to_array_fn(audio_segment, sampling_rate)

            audio_segment = speech_file_to_array_fn1(f"temp.wav", sampling_rate)

            # print("333")
            # 获取性别识别结果
            gender = predict_gender(audio_segment, sampling_rate)
            # print("444")

            # 构建响应数据
            response_data[start_timestamp] = {
                'start_time': start_timestamp,
                'end_time': end_timestamp,
                'text': subtitle.content,
                'gender': gender
            }

            print(response_data[start_timestamp])
        except Exception as e:
            print(f"gender 识别发生错误: {e}")       

    return jsonify(response_data)

@app.route('/gender_recognition', methods=['POST'])
def gender_recognition():
    wav_file = request.files['wav']
    srt_file = request.files['srt']

    # 读取srt文件内容
    srt_content = srt_file.read().decode('utf-8')
    subtitles = list(srt.parse(srt_content))

    # 读取wav音频数据
    audio = AudioSegment.from_wav(io.BytesIO(wav_file.read()))

    sampling_rate = feature_extractor.sampling_rate
    new_srt = []

    # 处理每个字幕时间段
    for subtitle in subtitles:        
        print(subtitle)
        try:
            start_time_ms = int(subtitle.start.total_seconds() * 1000)
            end_time_ms = int(subtitle.end.total_seconds() * 1000)

            if (end_time_ms - start_time_ms) < 1000:
                end_time_ms = start_time_ms + 1000

            # 格式化时间戳为HH:mm:ss,sss
            start_timestamp = format_timestamp(subtitle.start)
            end_timestamp = format_timestamp(subtitle.end)

            # 根据时间段分割音频
            audio_segment = slice_wav_by_timestamp(audio, start_time_ms, end_time_ms, start_timestamp)
            audio_segment = speech_file_to_array_fn1(f"temp.wav", sampling_rate)

            # 获取性别识别结果
            gender = predict_gender(audio_segment, sampling_rate)
            print(f'Gender: ${gender}')

            # 根据识别的性别添加前缀
            prefix = '[M]' if gender == 'M' else '[F]'
            new_content = prefix + subtitle.content

            # 修改字幕内容
            subtitle.content = new_content

            # 将修改后的字幕添加到新的SRT列表
            new_srt.append(subtitle)
        except Exception as e:
            new_srt.append(subtitle)
            print(f"gender 识别发生错误: {e}")       

    # 将修改后的字幕重新转换为SRT格式
    new_srt_content = srt.compose(new_srt)

    return jsonify({"srt": new_srt_content})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)