param(
    [ValidateSet('Start', 'Stop', 'Restart', 'Status', 'Wait')]
    [string] $Command = 'Start',

    [string] $Distro = '',
    [string] $RepoPath = '/mnt/c/Users/adyba/src/lucebox-hub',
    [int] $WaitSeconds = 300
)

$ErrorActionPreference = 'Stop'

$scriptPath = "$RepoPath/scripts/lucebox-gemma4-4090.sh"
$wslArgsPrefix = @()
if ($Distro -ne '') {
    $wslArgsPrefix += @('-d', $Distro)
}

function Invoke-LuceboxWsl {
    param([string] $Bash)
    & wsl.exe @wslArgsPrefix -e bash -lc $Bash
}

function New-WslArgumentLine {
    param([string] $Bash)

    $parts = @()
    $parts += $wslArgsPrefix
    $parts += @('-e', 'bash', '-lc', $Bash)

    ($parts | ForEach-Object {
        $part = [string] $_
        if ($part -match '[\s"]') {
            '"' + ($part -replace '"', '\"') + '"'
        } else {
            $part
        }
    }) -join ' '
}

switch ($Command) {
    'Start' {
        Invoke-LuceboxWsl "rm -f `"`$HOME/lucebox-runs/lucebox-gemma4-mtp-server.pid`""
        $bash = "chmod +x '$scriptPath'; exec '$scriptPath' run"
        $startArgs = New-WslArgumentLine $bash
        $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList $startArgs -PassThru -WindowStyle Hidden
        "winpid=$($proc.Id)"
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; '$scriptPath' wait $WaitSeconds"
    }
    'Stop' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; '$scriptPath' stop"
    }
    'Restart' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; '$scriptPath' stop || true"
        $bash = "chmod +x '$scriptPath'; exec '$scriptPath' run"
        $startArgs = New-WslArgumentLine $bash
        $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList $startArgs -PassThru -WindowStyle Hidden
        "winpid=$($proc.Id)"
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; '$scriptPath' wait $WaitSeconds"
    }
    'Status' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; '$scriptPath' status"
    }
    'Wait' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; '$scriptPath' wait $WaitSeconds"
    }
}
