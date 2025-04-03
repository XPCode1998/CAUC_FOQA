from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from math import radians, sin, cos, sqrt, atan2

import hashlib


# Create your views here.
# Create your views here.
# 用户注册页面
def register(request):
    # GET请求方式：打开用户注册页面
    if request.method == 'GET':
        info = '请输入您的用户名和密码'
        print('注册')
        return render(request, 'auth/register.html', {'info':info})
     # POST请求方式：提交注册请求
    elif request.method == 'POST':
        # 用户名
        username = request.POST.get('username','')
        # 密码
        password_1 = request.POST.get('password_1','')
        # 确认密码
        password_2 = request.POST.get('password_2','')
        # 比较密码与确认密码，如果不一致，返回注册错误信息
        if password_1 != password_2:
            info = '两次密码输入不一致'
            return render(request, 'auth/register.html', {'info':info})
        if User.objects.filter(username=username):
            info = '用户名已注册'
            return render(request, 'auth/register.html', {'info':info})
        else:
            # 哈希算法，对密码进行加密
            m = hashlib.md5()
            m.update(password_1.encode())
            password_m = m.hexdigest()
            # 创建新的用户
            try:
                d = dict(username=username, password=password_m)
                user = User.objects.create_user(**d)
                return HttpResponseRedirect('login')
            # 若在数据库中插入失败，则返回注册错误信息
            except Exception as e:
                info = '注册失败'
                return render(request, 'auth/register.html', {'info':info})


# 用户登录页面
def login(request):
    # GET请求方式：打开用户登录页面
    if request.method == 'GET':
        info = '请输入用户名和密码'
        # 返回用户登录页面
        return render(request, 'auth/login.html', {'info':info})
    # POST请求方式：提交登录请求
    elif request.method == 'POST':
         # 用户名
        username = request.POST.get('username','')
        # 密码
        password = request.POST.get('password','')
        # 去用户信息表中查询是否存在该用户
        if User.objects.filter(username=username):
            # 密码加密，与数据库中加密密码进行比对
            m = hashlib.md5()
            m.update(password.encode())
            print('密码',m.hexdigest())
            user = authenticate(username=username, password = m.hexdigest())
            if user:
                if user.is_active :
                    # 创建session对象
                    request.session['username'] = username
                    request.session['uid'] = user.id
                    # 首页跳转对象
                    resp = HttpResponseRedirect('qar_upload')
                    # 如果用户勾选了“记住我”
                    if 'remember' in request.POST:
                        # 创建Cookies对象，有效期3天
                        resp.set_cookie('username', username, 3600 * 24 * 3)
                        resp.set_cookie('uid', user.id, 3600 * 24 * 3)
                    # 登录成功，跳转至首页
                    print('登录成功')
                    return resp
            else:
                info = '用户名或密码错误, 请重新输入'
        else:
            info = '用户名不存在'
        return render(request, 'auth/login.html', {'info':info})