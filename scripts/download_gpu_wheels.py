"""按官方锁清单四路断点下载GPU依赖，完整SHA256通过后才发布wheel文件。"""

import concurrent.futures
import datetime
import hashlib
import http.client
import json
import os
import shutil
import threading
import time
import urllib.parse
from pathlib import Path

items=json.loads(Path('reports/gpu-wheel-lock.json').read_text())
root=Path('.cache/gpu-wheelhouse')
root.mkdir(parents=True,exist_ok=True)
part_size=8*1024*1024
lock=threading.Lock()
connections=threading.local()
state_path=Path('.cache/gpu-download-state.json')
signature=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()
state=json.loads(state_path.read_text()) if state_path.exists() else {'signature':signature,'done':{}}
if state['signature']!=signature:
    raise RuntimeError('依赖锁与断点身份不一致')
if shutil.disk_usage(root).free<18*1024**3:
    raise RuntimeError('磁盘余量不足18GiB，停止GPU依赖下载')
started=time.monotonic()
started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
downloaded_bytes=0

def sha256(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        while block:=handle.read(1024*1024):
            value.update(block)
    return value.hexdigest()

def save_state():
    temp=state_path.with_suffix('.tmp')
    temp.write_text(json.dumps(state))
    os.replace(temp,state_path)

jobs=[]
for item in items:
    name=urllib.parse.unquote(urllib.parse.urlparse(item['url']).path.rsplit('/',1)[1])
    if Path(name).name!=name or not name.endswith('.whl'):
        raise ValueError('wheel文件名无效')
    item['filename']=name
    final=root/name
    if final.exists():
        if final.stat().st_size!=item['size'] or sha256(final)!=item['sha256']:
            raise ValueError('已有wheel校验失败：'+item['name'])
        continue
    partial=root/(name+'.partial')
    if not partial.exists():
        with partial.open('wb') as handle:
            handle.truncate(item['size'])
    elif partial.stat().st_size!=item['size']:
        raise ValueError('临时wheel尺寸与锁记录不一致')
    complete=set(state['done'].get(name,[]))
    for index,start in enumerate(range(0,item['size'],part_size)):
        if index not in complete:
            jobs.append((index,item,start,min(item['size']-1,start+part_size-1)))
jobs.sort(key=lambda value:value[0])

def fetch(job):
    global downloaded_bytes
    index,item,start,end=job
    name=item['filename']
    parsed=urllib.parse.urlparse(item['url'])
    if parsed.scheme!='https' or parsed.netloc not in ('download-r2.pytorch.org','files.pythonhosted.org'):
        raise ValueError('下载域名不在已核验的官方来源中')
    position=0
    length=end-start+1
    for attempt in range(5):
        connection=None
        try:
            if position==length:
                break
            if not hasattr(connections,'by_host'):
                connections.by_host={}
            connection=connections.by_host.get(parsed.netloc)
            if connection is None:
                connection=http.client.HTTPSConnection(parsed.netloc,timeout=30)
                connections.by_host[parsed.netloc]=connection
            connection.request('GET',parsed.path+('?' + parsed.query if parsed.query else ''),
                               headers={'Range':f'bytes={start+position}-{end}','Connection':'keep-alive'})
            request_started=time.monotonic()
            with connection.getresponse() as response,(root/(name+'.partial')).open('r+b',buffering=0) as handle:
                if response.status in (403,429):
                    raise PermissionError(f'官方来源返回HTTP{response.status}')
                expected=f'bytes {start+position}-{end}/{item["size"]}'
                if response.status!=206 or response.headers.get('Content-Range')!=expected:
                    raise ValueError('官方响应分段范围不符')
                handle.seek(start+position)
                while position<length:
                    block=response.read(min(256*1024,length-position))
                    if not block:
                        raise EOFError('响应提前结束')
                    handle.write(block)
                    position+=len(block)
                    with lock:
                        downloaded_bytes+=len(block)
                    if time.monotonic()-request_started>120:
                        raise TimeoutError('单次分段请求超过120秒')
                os.fsync(handle.fileno())
            break
        except PermissionError:
            raise
        except Exception as exc:
            if connection is not None:
                connection.close()
                connections.by_host.pop(parsed.netloc,None)
            print(f'重试：{item["name"]} 第{index}段 第{attempt+1}次 {type(exc).__name__}',flush=True)
            time.sleep(min(2**attempt,8))
    if position!=length:
        raise RuntimeError(f'分段有限重试失败：{item["name"]} {index}')
    with lock:
        state['done'].setdefault(name,[]).append(index)
        save_state()
        print(f'本轮新增MB={downloaded_bytes/1e6:.1f}，实际MB/s={downloaded_bytes/1e6/(time.monotonic()-started):.2f}',flush=True)

errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(fetch,job) for job in jobs]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            errors.append(str(exc))
            print(str(exc),flush=True)
if not errors:
    for item in items:
        final=root/item['filename']
        partial=root/(item['filename']+'.partial')
        path=final if final.exists() else partial
        if sha256(path)!=item['sha256']:
            errors.append('完整wheel摘要失败：'+item['name'])
            continue
        if partial.exists():
            os.replace(partial,final)
        print('完整SHA256通过：'+item['name'],flush=True)
report={'started_at_utc':started_at,'ended_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source':'服务器直连官方PyTorch/PyPI；4线程复用TLS，无大文件SSH传输',
        'new_downloaded_bytes':downloaded_bytes,'seconds':time.monotonic()-started,
        'success':not errors,'errors':errors}
Path('.cache/gpu-download-window.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False),flush=True)
if errors:
    raise SystemExit(1)
