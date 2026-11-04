# 在 Windows 终端保持运行；关闭终端或按 Ctrl+C 后转发结束。
param(
    [string]$RemoteHost = '10.103.69.210',
    [string]$RemoteUser = 'luojinghong',
    [ValidateRange(1024, 65535)][int]$WebPort = 18080,
    [ValidateRange(1024, 65535)][int]$ApiPort = 19380
)

$ErrorActionPreference = 'Stop'
if ($WebPort -eq $ApiPort) {
    throw '网页和 API 的本机端口必须不同。'
}
if (-not (Get-Command ssh.exe -ErrorAction SilentlyContinue)) {
    throw '未找到 Windows OpenSSH 客户端 ssh.exe。'
}

# 使用独立配置，避免重复建立用户 SSH 配置中的反向代理端口。
# 身份验证仍由 OpenSSH 默认密钥和已有 known_hosts 完成，不关闭主机校验。
Write-Host "保持本窗口运行，连接成功后访问 http://127.0.0.1:$WebPort"
& ssh.exe -F NUL -N -T `
    -o BatchMode=yes `
    -o ExitOnForwardFailure=yes `
    -o ServerAliveInterval=30 `
    -o ServerAliveCountMax=3 `
    -L "127.0.0.1:${WebPort}:127.0.0.1:18080" `
    -L "127.0.0.1:${ApiPort}:127.0.0.1:19380" `
    "$RemoteUser@$RemoteHost"
exit $LASTEXITCODE
