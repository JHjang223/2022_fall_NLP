import os

directory_list = ['./ckpt/emotion/kcelectra-base-v3-ISDS-ckpt/test/',
                  './ckpt/emotion/kcelectra-base-v3-emotion-ckpt_attn_size_500/test/']

acc_ = []
acc_MLP_500 = []

CNT = 0
for i in directory_list:
    CNT += 1
    for j in os.listdir(i):
        tmp =os.path.join(i, j)
        with open(tmp, 'r', encoding='utf-8') as f:
            acc_tmp = f.readlines()[0].strip()
            if CNT == 1:
                acc_.append(float(acc_tmp[6:]))
            elif CNT ==2:
                acc_MLP_500.append(float(acc_tmp[6:]))

print(max(acc_))
print(acc_.index(max(acc_)))
print(max(acc_MLP_500))
print(acc_MLP_500.index((max(acc_MLP_500))))


