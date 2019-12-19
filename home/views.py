from django.shortcuts import render, HttpResponse
from keras.models import load_model
from keras.preprocessing import sequence
from ckiptagger import WS, POS, NER
import json
import re
import os
import h5py
from opencc import OpenCC

# Create your views here.

def home(request):
    return render(request, 'home/home.html')

def test(request):
    country = ''#{'country': 'hey'} 
    if request.method == 'POST':
        text=request.POST.get('input')
        print(text)

        cc = OpenCC('s2tw')

        text = cc.convert(text)

        t = text

        with open("./file/Encode.json", 'r') as j:
            encode_dict = json.load(j)
            j.close

        model = load_model('./file/my_model.h5')
        #  text = input()

        text = text.replace('+', '').replace('-', '').replace('‘', '').replace('’', '').replace('\t', '').replace('\xa0','').replace('\n','').replace(' ','').replace('\u3000','').replace('[^\w\s]','').replace('“',"").replace('”',"").replace('／',"").replace('《','').replace('》','').replace('，','').replace('。','').replace('「','').replace('」','').replace('（','').replace('）','').replace('！','').replace('？','').replace('、','').replace('▲','').replace('…','').replace('：','').replace('；','').replace('—','').replace('●','').replace('■','').replace('【','').replace('】','').replace('(','').replace(')','').replace('〔','').replace('〕','').replace('!','').replace('?','').replace('︹','').replace('︺','')

        ws = WS("./file/data")
        ws_results = ws([text])
        del ws
        seg = ' '.join(ws_results[0])

        test = list()
        text = seg.split(' ')
        x = list()
        x.append(test)

        for voc in text:
            if voc in encode_dict:
                num = encode_dict[voc]

                if num<5000:
                    test.append(num)

                else:
                    test.append(0)

            else:
                test.append(0)

        x = sequence.pad_sequences(x, maxlen=500)
        labels = [int(round(x[0])) for x in model.predict(x) ]
        ans = labels[0]

        print(labels[0])

        if ans:
            country = '臺灣' # {'country': '臺灣'}

        else:
            country = '中國' # {'country': '中國'}

    print(country)

    return render(request,'home/home.html', locals())



