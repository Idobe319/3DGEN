# Run TRELLIS verification and extract the key log snippet
# Usage (PowerShell, from repo root):
#   .\scripts\run_trellis_verify.ps1

param(
    [string]$CondaEnv = 'trellis',
    [string]$InputPath = 'samples/example.jpg',
    [string]$Weights = '.\tsr\model.ckpt',
    [string]$OutDir = 'workspace_trellis_real',
    [int]$ContextLines = 30
)

# ensure outdir
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
$log = Join-Path $OutDir 'run.log'

# verify input path exists; if not, try to find a sample image under samples/
if (-not (Test-Path $InputPath)) {
    Write-Host "Default input '$InputPath' not found, searching for any image under 'samples/'..."
    $candidate = Get-ChildItem -Path 'samples' -Include *.jpg,*.jpeg,*.png -Recurse -File | Where-Object { $_.Length -gt 1024 } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($candidate) {
        $InputPath = $candidate.FullName
        Write-Host "Found candidate input: $InputPath"
    } else {
        Write-Host "No sample image found in 'samples/'. Please provide --InputPath to the script or put a test image in samples/" -ForegroundColor Red
        exit 2
    }
}

# Remove any injected attention backend (let TRELLIS pick a compatible one) and reduce TF noise
Remove-Item Env:TRELLIS_ATTENTION -ErrorAction SilentlyContinue
Remove-Item Env:ATTN_BACKEND -ErrorAction SilentlyContinue
$env:FLASH_ATTENTION = '0'
$env:USE_FLASH_ATTENTION = '0'
$env:USE_FLASH_ATTN = '0'
$env:XFORMERS_DISABLED = '1'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
$env:TF_ENABLE_ONEDNN_OPTS = '0'
# Prefer local model folder if present (avoid passing a single .ckpt path)
$localModelDir = Resolve-Path -LiteralPath '.\tsr\TRELLIS-image-large' -ErrorAction SilentlyContinue
if ($localModelDir) {
    Write-Host ("Local TRELLIS model dir found: {0} - will pass as --trellis-url" -f $localModelDir.Path)
} else {
    Write-Host 'No local TRELLIS model dir found at tsr\TRELLIS-image-large; will fall back to using --trellis-weights (if present) or HF repo.'
}

# Prefer running the env python directly (avoid activation/VC prints) when possible
$condaBase = (& conda info --base) -join '' | ForEach-Object { $_.Trim() }
$envPython = ''
if ($condaBase) {
    $possible = Join-Path $condaBase "envs\$CondaEnv\python.exe"
    if (Test-Path $possible) { $envPython = $possible }
}

try {
    if ($envPython) {
        Write-Host "Using python executable from conda env: $envPython"
        $exe = $envPython
        if ($localModelDir) {
            $procArgs = @('-u', 'run_local.py', '--input', $InputPath, '--engine', 'trellis', '--trellis-url', $localModelDir.Path, '--fast-preview', '--quality', 'low', '-o', $OutDir)
        } else {
            $procArgs = @('-u', 'run_local.py', '--input', $InputPath, '--engine', 'trellis', '--trellis-weights', $Weights, '--fast-preview', '--quality', 'low', '-o', $OutDir)
        }
    } else {
        Write-Host "Falling back to 'conda run -n $CondaEnv' (env python not discovered automatically)"
        $exe = 'conda'
        if ($localModelDir) {
            $procArgs = @('run', '-n', $CondaEnv, 'python', '-u', 'run_local.py', '--input', $InputPath, '--engine', 'trellis', '--trellis-url', $localModelDir.Path, '--fast-preview', '--quality', 'low', '-o', $OutDir)
        } else {
            $procArgs = @('run', '-n', $CondaEnv, 'python', '-u', 'run_local.py', '--input', $InputPath, '--engine', 'trellis', '--trellis-weights', $Weights, '--fast-preview', '--quality', 'low', '-o', $OutDir)
        }
    }
    Write-Host "Command: $exe $($procArgs -join ' ')"
    & $exe @procArgs 2>&1 | Tee-Object -FilePath $log
    $exit = $LASTEXITCODE
} catch {
    Write-Host "Execution failed: $_"
    $exit = 1
}

Write-Host "Log saved to: $log"

# Extract context around key patterns
$patterns = 'Loading TRELLIS pipeline', 'Running TRELLIS inference', 'TRELLIS failed', 'falling back to TripoSR'
$found = $false
foreach ($p in $patterns) {
    $foundMatches = Select-String -Path $log -Pattern $p -Context $ContextLines,$ContextLines -SimpleMatch
    if ($foundMatches) {
        $found = $true
        Write-Host ""
        Write-Host ("--- Snippets around pattern: '{0}' ---" -f $p)
        Write-Host ""
        foreach ($m in $foundMatches) {
            # Print matched block
            $pre = $m.Context.PreContext
            $post = $m.Context.PostContext
            foreach ($line in $pre) { Write-Host $line }
            Write-Host $m.Line
            foreach ($line in $post) { Write-Host $line }
            Write-Host '----------------------------------------'
        }
    }
}

if (-not $found) {
    Write-Host "No matching TRELLIS markers found in log. Printing head of log to help debugging..." -ForegroundColor Yellow
    Get-Content -Path $log -TotalCount 200 | ForEach-Object { Write-Host $_ }
}

if ($exit -ne 0) {
    Write-Host "Verification command exited with code $exit" -ForegroundColor Yellow
} else {
    Write-Host 'Verification command completed (exit code 0). If TRELLIS ran, the snippets above will show "Loading TRELLIS pipeline" and "Running TRELLIS inference".' -ForegroundColor Green
}
