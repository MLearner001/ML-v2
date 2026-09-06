import re
with open("phase2_experimental/train_bi_lstm.py", "r") as f:
    text = f.read()

text = text.replace("Karena target variabel (Duration_Sec) sudah di log1p", "Karena target variabel (Target_Feedrate) sudah di log1p")
with open("phase2_experimental/train_bi_lstm.py", "w") as f:
    f.write(text)
