# 1.FastAPI基础入门

## 1.3第一个FastAPI程序

```python
from fastapi import FastAPI

#创建FastAPI实例
app = FastAPI()

@app.get("/")
async def root(): # 定义一个根路由，返回一个字典。async 表示异步。
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
```

运行命令：

```
cd C:\Users\SONGQI\Desktop\FastAPI
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

其中，`uvicorn` 是一个用来运行 FastAPI 项目的 Web 服务器。

`--reload` ：开发模式。你修改代码后，服务器会自动重启。

`--host 127.0.0.1`：只允许本机访问，也就是只能在你自己的电脑浏览器里打开。

`--port 8000`：使用 8000 端口，所以访问地址是

```
http://127.0.0.1:8000/
```

 

[FastAPI - Swagger UI](http://127.0.0.1:8000/docs#/)

