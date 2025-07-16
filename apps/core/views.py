from django.shortcuts import render,redirect

# Create your views here.
def index(request):
    # 首页页面跳转逻辑
    # 若用户已登录，浏览器中存放着Cookies或session值，则用户免登陆直接跳转首页
    s_username = request.session.get('username')
    s_uid = request.session.get('uid')
    c_username = request.COOKIES.get('username')
    c_uid = request.COOKIES.get('uid')
    if (s_username and s_uid) or (c_username and c_uid):
        return redirect('flight_monitor/flight_preview')
    # 若用户未登录，则跳转到登录页面
    else:
        info = '请先登录'
        return render(request, 'auth/login.html', {'info': info})