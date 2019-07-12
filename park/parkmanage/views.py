import sys,os
import tensorflow as tf
from django.forms import fields
from django.shortcuts import render,redirect
from django.http import HttpResponse
from .models import Car
from django import forms
import django.utils.timezone as timezone
import time,datetime,math
import positioning
# # Create your views here.

def hello(request):
    return render(request,'main.html')

def test(request):
    return render(request,'main.html')

def car_add(request):
    cnum = request.GET.get('cnum')
    if cnum == "":
        return HttpResponse('无车辆')
    cexist = Car.objects.filter(cname=cnum).first()
    if cexist == None:
        new_car = Car(cname=cnum)
        new_car.save()
        return HttpResponse('车辆取卡成功！')
    car1 = Car.objects.get(cname=cnum)
    if car1.statime == car1.endtime:
        car1.statime = timezone.now()
        car1.save()
        return HttpResponse('车辆取卡成功！')
    return HttpResponse('车辆已添加至停车场，请勿重复操作！')


def car_account(request):
    cnum = request.GET.get('cnum')
    cexist = Car.objects.filter(cname=cnum).first()
    if cexist == None:
        return HttpResponse('停车场无此车辆记录，无法结账!')
    car1 = Car.objects.get(cname=cnum)
    if car1.statime == car1.endtime:
        return HttpResponse('此车辆未进入停车场！')
    car1.endtime = timezone.now()
    car1.save()
    pass
    #datetime转换为时间戳
    etime = car1.endtime.strftime("%Y-%m-%d %H:%M:%S")
    stime = car1.statime.strftime("%Y-%m-%d %H:%M:%S")
    ee = time.mktime(time.strptime(etime,"%Y-%m-%d %H:%M:%S"))
    ss = time.mktime(time.strptime(stime, "%Y-%m-%d %H:%M:%S"))
    cost = str(math.ceil((ee-ss)/3600)*2)#向上取整,模拟2元1小时
    car1.costs += float(cost)
    car1.save()
    return HttpResponse("起始时间："+stime+"<br>结束时间："+etime
                        +"<br>花费："+cost+" 元")

def finish(request):
    cnum = request.GET.get('cnum')
    cexist = Car.objects.filter(cname=cnum).first()
    if cexist == None:
        return HttpResponse('没有对应车辆或者没有识别出车辆')
    car1 = Car.objects.get(cname=cnum)
    if car1.statime == datetime.datetime(1970, 1, 1, 0, 0, 0):
        return HttpResponse("车辆尚未入库，请先点击计费按钮!")
    if car1.endtime == datetime.datetime(1970, 1, 1, 0, 0, 0):
        return HttpResponse("车辆尚未结账，请先点击结账按钮!")
    car1.statime = datetime.datetime(1970, 1, 1, 0, 0, 0)
    car1.endtime = datetime.datetime(1970, 1, 1, 0, 0, 0)
    car1.save()
    return HttpResponse("交易完成")

class UploadForm(forms.Form):
    file = fields.FileField()

def getCarNum(request):
    image = request.FILES.get("imgfile",None)
    if not image:
        return HttpResponse("上传失败")
    image_data = [image.file, image.field_name, image.name, image.content_type,
                  image.size, image.charset, image.content_type_extra]
    print(image_data)
    file_name = './static/img/'+ 'temp' + '.' + image.name.split('.')[-1]
    # 构造文件名以及文件路径
    if image.name.split('.')[-1] not in ['jpeg', 'jpg', 'png', 'gif']:
        return HttpResponse('输入文件有误')
    try:
        pass
        with open(file_name, 'wb+') as f:
            f.write(image.read())
    except Exception as e:
        print(e)
    positioning.get_img_data('static/img/temp.jpg')

    from license_province import predict_province
    predict_province()
    from license_digits import predict_digits
    predict_digits()

    def output_license():
        f2 = open("parkmanage/out.txt", 'r', encoding="utf-8")
        license_data = f2.read()
        f2.close()
        print(license_data)
        return license_data

    license_data = output_license()

    return HttpResponse(str(license_data))

