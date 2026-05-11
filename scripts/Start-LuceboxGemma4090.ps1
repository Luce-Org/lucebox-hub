param(
    [ValidateSet('Start', 'Stop', 'Restart', 'Status', 'Wait')]
    [string] $Command = 'Start',

    [string] $Distro = '',
    [string] $RepoPath = '/mnt/c/Users/adyba/src/lucebox-hub',
    [int] $WaitSeconds = 300,
    [int] $ContextSize = 40960,
    [int] $DraftContextSize = 2048,
    [int] $DraftNMax = 4,
    [int] $BatchSize = 2048,
    [int] $UBatchSize = 512,
    [string] $CacheTypeK = 'q8_0',
    [string] $CacheTypeV = 'q8_0',
    [string] $DraftCacheTypeK = '',
    [string] $DraftCacheTypeV = '',
    [string] $CacheRam = '0'
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

$effectiveDraftCacheTypeK = if ($DraftCacheTypeK -ne '') { $DraftCacheTypeK } else { $CacheTypeK }
$effectiveDraftCacheTypeV = if ($DraftCacheTypeV -ne '') { $DraftCacheTypeV } else { $CacheTypeV }

function ConvertTo-BashSingleQuoted {
    param([string] $Value)
    "'" + ($Value -replace "'", "'\"'\"'") + "'"
}

function Get-LuceboxEnvPrefix {
    $pairs = [ordered] @{
        LUCEBOX_GEMMA4_CTX_SIZE = [string] $ContextSize
        LUCEBOX_GEMMA4_DRAFT_CTX_SIZE = [string] $DraftContextSize
        LUCEBOX_GEMMA4_DRAFT_N_MAX = [string] $DraftNMax
        LUCEBOX_GEMMA4_BATCH_SIZE = [string] $BatchSize
        LUCEBOX_GEMMA4_UBATCH_SIZE = [string] $UBatchSize
        LUCEBOX_GEMMA4_CACHE_TYPE_K = $CacheTypeK
        LUCEBOX_GEMMA4_CACHE_TYPE_V = $CacheTypeV
        LUCEBOX_GEMMA4_DRAFT_CACHE_TYPE_K = $effectiveDraftCacheTypeK
        LUCEBOX_GEMMA4_DRAFT_CACHE_TYPE_V = $effectiveDraftCacheTypeV
        LUCEBOX_GEMMA4_CACHE_RAM = $CacheRam
    }
    ($pairs.GetEnumerator() | ForEach-Object {
        "$($_.Key)=$(ConvertTo-BashSingleQuoted ([string] $_.Value))"
    }) -join ' '
}

$envPrefix = Get-LuceboxEnvPrefix

switch ($Command) {
    'Start' {
        Invoke-LuceboxWsl "rm -f `"`$HOME/lucebox-runs/lucebox-gemma4-mtp-server.pid`""
        $bash = "chmod +x '$scriptPath'; $envPrefix exec '$scriptPath' run"
        $startArgs = New-WslArgumentLine $bash
        $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList $startArgs -PassThru -WindowStyle Hidden
        "winpid=$($proc.Id)"
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' wait $WaitSeconds"
    }
    'Stop' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' stop"
    }
    'Restart' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' stop || true"
        $bash = "chmod +x '$scriptPath'; $envPrefix exec '$scriptPath' run"
        $startArgs = New-WslArgumentLine $bash
        $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList $startArgs -PassThru -WindowStyle Hidden
        "winpid=$($proc.Id)"
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' wait $WaitSeconds"
    }
    'Status' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' status"
    }
    'Wait' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' wait $WaitSeconds"
    }
}
