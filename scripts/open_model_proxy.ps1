# 运行期间保持此窗口打开。仅在服务器回环地址提供本机已有的代理。
param(
    [string]$RemoteHost = '10.103.69.210',
    [string]$RemoteUser = 'luojinghong',
    [ValidateRange(1, 65535)][int]$LocalProxyPort = 7890,
    [ValidateRange(1024, 65535)][int]$RemoteProxyPort = 17893
)
$ErrorActionPreference = 'Stop'
$probe = [System.Net.Sockets.TcpClient]::new()
try {
    $probe.Connect('127.0.0.1', $LocalProxyPort)
} finally {
    $probe.Dispose()
}
Write-Host "转发服务器回环端口 $RemoteProxyPort 到本机代理 $LocalProxyPort。按 Ctrl+C 结束。"
while ($true) {
    & ssh.exe -F NUL -N -T -o BatchMode=yes -o ConnectTimeout=8 -o ExitOnForwardFailure=yes `
        -o ServerAliveInterval=10 -o ServerAliveCountMax=3 `
        -R "127.0.0.1:${RemoteProxyPort}:127.0.0.1:${LocalProxyPort}" "$RemoteUser@$RemoteHost"
    Write-Host "SSH 连接已退出。若端口已有健康转发，可关闭本窗口；否则 5 秒后重试。"
    Start-Sleep -Seconds 5
}

