from django.shortcuts import render, redirect
import pickle
import sklearn
import numpy as np
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)

# Create your views here.


def index(request):
    return render(request, 'index.html')


def WinePredict(request):
    if request.method == "POST":
        fa = request.POST.get('fixedacidity')
        c = request.POST.get('citric')
        s = request.POST.get('sulphur')
        sul = request.POST.get('sulphate')
        a = request.POST.get('alcohol')

        with open('model_pkl', 'rb') as f:

            lr = pickle.load(f)
        output = lr.predict(transformer.fit_transform(
            [[float(fa), float(c), float(s), float(sul), float(a)]]))
        res = ''
        if output[0] <= 4:
            res = 'Poor'
        elif output[0] >= 4 and output[0] <= 5:
            res = 'Average'
        else:
            res = "Strong"
        print(res)
        context = {
            'data': output[0],
            'q': res,
        }
        return render(request, 'output.html', context)

    return render(request, 'predict.html')


def Output(request):
    return render(request, 'output.html')
