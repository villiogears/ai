from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
import json
import os 

app = Flask(__name__)

# 学習データの準備
data = [
      {"input": "おはようございます", "output": "おはようございます！"},
  {"input": "こんにちは", "output": "こんにちは！"},
  {"input": "こんばんは", "output": "こんばんは。"},
  {"input": "おやすみなさい", "output": "おやすみなさい。"},
  {"input": "いってきます", "output": "いってらっしゃい。"},
  {"input": "ただいま", "output": "おかえりなさい。"},
  {"input": "いただきます", "output": "どうぞ、めしあがれ。"},
  {"input": "ごちそうさまでした", "output": "お粗末さまでした。"},
  {"input": "すみません", "output": "はい、なんでしょうか？"},
  {"input": "ありがとうございます", "output": "どういたしまして。"},
  {"input": "お久しぶりです", "output": "本当に、お久しぶりですね！"},
  {"input": "お元気ですか？", "output": "はい、おかげさまで元気です。"},
  {"input": "最近どうですか？", "output": "まあまあです。そちらはいかがですか？"},
  {"input": "お疲れ様です", "output": "ありがとうございます。あなたもお疲れ様です。"},
  {"input": "頑張ってください", "output": "ありがとうございます！頑張ります。"},
  {"input": "応援しています", "output": "ありがとうございます。心強いです。"},
  {"input": "助かります", "output": "お役に立てて嬉しいです。"},
  {"input": "気にしないでください", "output": "ありがとうございます。そうさせていただきます。"},
  {"input": "了解しました", "output": "ありがとうございます。助かります。"},
  {"input": "承知いたしました", "output": "ありがとうございます。それでは、そのように手配いたします。"},
  {"input": "少々お待ちください", "output": "はい、かしこまりました。"},
  {"input": "お待たせしました", "output": "ありがとうございます。"},
  {"input": "どうぞ", "output": "ありがとうございます。いただきます。"},
  {"input": "遠慮なく", "output": "では、お言葉に甘えて。"},
  {"input": "ごゆっくりどうぞ", "output": "ありがとうございます。"},
  {"input": "お気をつけて", "output": "はい、ありがとうございます。"},
  {"input": "またね", "output": "はい、また。"},
  {"input": "さようなら", "output": "さようなら。"},
  {"input": "おめでとうございます", "output": "ありがとうございます。"},
  {"input": "誕生日おめでとう", "output": "ありがとう！"},
  {"input": "明けましておめでとうございます", "output": "明けましておめでとうございます！"},
  {"input": "今年もよろしくお願いします", "output": "こちらこそ、よろしくお願いします。"},
  {"input": "良い一日を", "output": "ありがとうございます。あなたも良い一日を。"},
  {"input": "良い週末を", "output": "ありがとうございます。良い週末をお過ごしください。"},
  {"input": "風邪をひかないでくださいね", "output": "ありがとうございます。気をつけます。"},
  {"input": "無理しないでくださいね", "output": "ありがとうございます。そうします。"},
  {"input": "何かお手伝いできることはありますか？", "output": "ありがとうございます。今のところ大丈夫です。"},
  {"input": "何か困ったことはありませんか？", "output": "今のところ大丈夫です。ありがとうございます。"},
  {"input": "ご心配おかけしてすみません", "output": "いいえ、お気になさらないでください。"},
  {"input": "大丈夫ですか？", "output": "はい、大丈夫です。ご心配ありがとうございます。"},
  {"input": "お邪魔します", "output": "いらっしゃいませ。どうぞ、お入りください。"},
  {"input": "失礼します", "output": "はい、どうぞ。"},
  {"input": "お手伝いしましょうか？", "output": "ありがとうございます。お願いします。"},
  {"input": "一緒にどうですか？", "output": "いいですね！"},
  {"input": "ちょっといいですか？", "output": "はい、構いませんよ。"},
  {"input": "あのう...", "output": "はい、なんでしょう？"},
  {"input": "すみません、道に迷ってしまいました", "output": "どちらへ行きたいのですか？"},
  {"input": "すみません、これ落としましたよ", "output": "ありがとうございます！"},
  {"input": "大丈夫？顔色が悪いよ", "output": "ちょっと疲れているだけです。ありがとう。"},
  {"input": "何かあった？", "output": "少し考え事をしています。"},
  {"input": "話聞こうか？", "output": "ありがとうございます。でも、大丈夫です。"},
  {"input": "気にしないで", "output": "はい、ありがとうございます。"},
  {"input": "ゆっくり休んでね", "output": "はい、ありがとうございます。そうします。"},
  {"input": "お大事にしてください", "output": "ありがとうございます。"},
  {"input": "また連絡します", "output": "はい、お待ちしています。"},
  {"input": "連絡ありがとう", "output": "どういたしまして。"},
  {"input": "楽しみですね", "output": "本当に！待ち遠しいです。"},
  {"input": "残念です", "output": "そうですね...仕方ないですね。"},
  {"input": "お疲れ様でした", "output": "ありがとうございます。"},
  {"input": "お先に失礼します", "output": "お疲れ様でした。"},
  {"input": "後ほど", "output": "はい、後ほど。"},
  {"input": "また明日", "output": "はい、また明日。"},
  {"input": "ご苦労様です", "output": "ありがとうございます。"},
  {"input": "ご迷惑をおかけしました", "output": "いいえ、大丈夫ですよ。"},
  {"input": "助かります", "output": "お役に立てて光栄です。"},
  {"input": "感謝します", "output": "こちらこそ、ありがとうございます。"},
  {"input": "素晴らしいですね", "output": "ありがとうございます！"},
  {"input": "すごいですね", "output": "ありがとうございます。"},
  {"input": "よくできました", "output": "ありがとうございます！"},
  {"input": "頑張ったね", "output": "ありがとうございます。"},
  {"input": "お上手ですね", "output": "ありがとうございます。まだまだですが..."},
  {"input": "センスがいいですね", "output": "ありがとうございます！嬉しいです。"},
  {"input": "面白いですね", "output": "ありがとうございます！そう言っていただけると嬉しいです。"},
  {"input": "楽しいですね", "output": "本当に！私もそう思います。"},
  {"input": "よかったですね", "output": "はい、本当に嬉しいです。"},
  {"input": "残念でしたね", "output": "はい...でも、また頑張ります。"},
  {"input": "仕方ないですね", "output": "そうですね..."},
  {"input": "そういうことですね", "output": "はい、そういうことです。"},
  {"input": "なるほど", "output": "ご理解いただけましたでしょうか？"},
  {"input": "わかりました", "output": "ありがとうございます。"},
  {"input": "教えてください", "output": "はい、何を知りたいですか？"},
  {"input": "あの...", "output": "はい？"},
  {"input": "もしもし", "output": "はい、もしもし。"},
  {"input": "どちら様ですか？", "output": "～です。"},
  {"input": "～さんはいらっしゃいますか？", "output": "少々お待ちください。"},
  {"input": "間違えました", "output": "失礼いたしました。"},
  {"input": "聞こえますか？", "output": "はい、聞こえますよ。"},
  {"input": "聞こえません", "output": "もう一度お願いします。"},
  {"input": "ちょっと遠いですね", "output": "もう少し近づきましょうか？"},
  {"input": "もう少しゆっくり話してください", "output": "はい、わかりました。"},
  {"input": "もう一度言ってください", "output": "はい、もう一度言いますね。"},
  {"input": "日本語は難しいですね", "output": "そうですね。でも、面白いですよ。"},
  {"input": "頑張って日本語を勉強します", "output": "応援しています！"},
  {"input": "いい天気ですね", "output": "本当に、気持ちがいいですね。"},
  {"input": "今日は暑いですね", "output": "ええ、汗ばみますね。"},
  {"input": "今日は寒いですね", "output": "はい、暖かくしてくださいね。"},
  {"input": "雨が降ってきましたね", "output": "傘を持っていますか？"},
  {"input": "傘を貸してください", "output": "どうぞ。"},
  {"input": "助かります", "output": "お役に立てて嬉しいです。"},
  {"input": "どうもありがとう", "output": "どういたしまして。"},
  {"input": "また連絡しますね", "output": "はい、楽しみにしています。"},
  {"input": "お会いできてよかったです", "output": "私も、お会いできて嬉しかったです。"},
  {"input": "またお会いしましょう", "output": "ぜひ、また。"},
  {"input": "お元気で", "output": "はい、あなたも。"},
  {"input": "さようなら", "output": "お元気で！"}
]

inputs = [item["input"] for item in data]
outputs = [item["output"] for item in data]

# 文字をインデックスに変換
char_to_index = {char: idx for idx, char in enumerate(sorted(set("".join(inputs + outputs))))}
index_to_char = {idx: char for char, idx in char_to_index.items()}

def text_to_sequence(text):
    return [char_to_index[char] for char in text]

def sequence_to_text(sequence):
    return "".join([index_to_char[idx] for idx in sequence])

# モデルの構築
max_len = max(len(seq) for seq in inputs)
X = tf.keras.preprocessing.sequence.pad_sequences([text_to_sequence(seq) for seq in inputs], maxlen=max_len, padding='post')
y = tf.keras.preprocessing.sequence.pad_sequences([text_to_sequence(seq) for seq in outputs], maxlen=max_len, padding='post')

y = tf.keras.utils.to_categorical(y, num_classes=len(char_to_index))

model = Sequential([
    Embedding(input_dim=len(char_to_index), output_dim=64, input_length=max_len),
    LSTM(128, return_sequences=True),
    Dense(len(char_to_index), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 学習
model.fit(X, y, epochs=500, verbose=0)

# 推論関数
def generate_response(prompt):
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([text_to_sequence(prompt)], maxlen=max_len, padding='post')
    pred = model.predict(input_seq)[0]
    response = sequence_to_text(np.argmax(pred, axis=-1))
    return response

# Flaskルート
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prompt', methods=['POST'])
def prompt():
    data = request.json
    user_input = data.get('input', '')
    response = generate_response(user_input)
    return jsonify({"input": user_input, "output": response})

if __name__ == '__main__':
    app.run(debug=True)


base_dir = "/home/manjaro/AI111"

# 仮想環境のフォルダを指定
venv_dir = os.path.join(base_dir, ".venv")
os.path.join(base_dir, "templates")

os.path.join("/usr/lib/libstdc++.so.6")
