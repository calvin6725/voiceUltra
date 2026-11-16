param([ValidateSet('index.html', 'video.html')][string]$Page = 'index.html')

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.runtime\Scripts\python.exe'

function Show-Failure([string]$Message) {
    try {
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show($Message, '声线实验室') | Out-Null
    } catch {
    }
}

function Test-Port([int]$Port) {
    $Client = New-Object Net.Sockets.TcpClient
    try {
        $Client.Connect('127.0.0.1', $Port)
        return $true
    } catch {
        return $false
    } finally {
        $Client.Close()
    }
}

try {
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        Show-Failure '本地声音克隆环境尚未安装，请先完成项目安装。'
        exit 1
    }

    $ChosenPort = $null
    $ReusedRunning = $false
    foreach ($CandidatePort in 8001..8010) {
        $Probe = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $CandidatePort)
        try {
            $Probe.Start()
            $Probe.Stop()
            $ChosenPort = $CandidatePort
            break
        } catch {
            try { $Probe.Stop() } catch {}
            try {
                $Existing = Invoke-RestMethod -Uri "http://127.0.0.1:$CandidatePort/api/clone/status" -TimeoutSec 2
                if ($Existing.engine -eq 'zipvoice') {
                    $ChosenPort = $CandidatePort
                    $ReusedRunning = $true
                    break
                }
            } catch {}
        }
    }
    if (-not $ChosenPort) { throw '没有找到可用的本地端口（8001-8010 均被占用）。' }

    if (-not $ReusedRunning) {
        $Arguments = @('"' + (Join-Path $ProjectRoot 'voice_server.py') + '"', '--port', "$ChosenPort")
        Start-Process -FilePath $PythonPath -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $ProjectRoot '.server-output.log') -RedirectStandardError (Join-Path $ProjectRoot '.server-error.log')
    }

    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 120; $Attempt++) {
        if (Test-Port $ChosenPort) { $Ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $Ready) { throw '本地声音克隆服务未能启动，请检查 .server-error.log。' }

    Start-Process "http://127.0.0.1:$ChosenPort/$Page"
    exit 0
} catch {
    Show-Failure "声线实验室启动失败：$($_.Exception.Message)"
    exit 1
}
