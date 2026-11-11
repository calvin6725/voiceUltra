param([ValidateSet('index.html', 'video.html')][string]$Page = 'index.html')

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $ProjectRoot '.runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonPath)) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show('本地声音克隆环境尚未安装，请先完成项目安装。', '声线实验室')
    exit 1
}

$ChosenPort = $null
foreach ($CandidatePort in 8001..8010) {
    try {
        $Existing = Invoke-RestMethod -Uri "http://localhost:$CandidatePort/api/clone/status" -TimeoutSec 1
        if ($Existing.engine -eq 'zipvoice') { $ChosenPort = $CandidatePort; break }
    } catch {}
    $Probe = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $CandidatePort)
    try {
        $Probe.Start()
        $ChosenPort = $CandidatePort
    } catch {
        continue
    } finally {
        $Probe.Stop()
    }
    if ($ChosenPort) {
        $Arguments = @('"' + (Join-Path $ProjectRoot 'voice_server.py') + '"', '--port', "$ChosenPort")
        Start-Process -FilePath $PythonPath -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $ProjectRoot '.server-output.log') -RedirectStandardError (Join-Path $ProjectRoot '.server-error.log')
        break
    }
}
if (-not $ChosenPort) { throw '没有找到可用的本地端口。' }
for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
    try {
        $Status = Invoke-RestMethod -Uri "http://localhost:$ChosenPort/api/clone/status" -TimeoutSec 1
        if ($Status.engine -eq 'zipvoice') {
            Start-Process "http://localhost:$ChosenPort/$Page"
            exit 0
        }
    } catch {}
    Start-Sleep -Milliseconds 500
}
throw '本地声音克隆服务未能启动，请检查 .server-error.log。'
