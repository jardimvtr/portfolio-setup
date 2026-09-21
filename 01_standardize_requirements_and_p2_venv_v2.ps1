param(
    [string]$PortfolioRoot = "C:\Users\vitor\Documents\Portfolio"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Step([string]$msg) {
    Write-Host ""
    Write-Host ("=" * 84) -ForegroundColor DarkGray
    Write-Host $msg -ForegroundColor Cyan
    Write-Host ("=" * 84) -ForegroundColor DarkGray
}

function Ok([string]$msg) {
    Write-Host "[OK] $msg" -ForegroundColor Green
}

function Info([string]$msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Yellow
}

$ProjectsRoot = Join-Path $PortfolioRoot "Projects"

if (-not (Test-Path -LiteralPath $ProjectsRoot)) {
    throw "Projects folder not found: $ProjectsRoot"
}

Set-Location -LiteralPath $PortfolioRoot

Step "1/4 - MOVE requirements.txt TO PROJECT ROOT"

foreach ($ProjectName in @("P1", "P2", "P3", "P4")) {
    $ProjectPath = Join-Path $ProjectsRoot $ProjectName

    if (-not (Test-Path -LiteralPath $ProjectPath)) {
        throw "Project folder not found: $ProjectPath"
    }

    $SetupDir = Join-Path $ProjectPath "setup"
    $OldReq = Join-Path $SetupDir "requirements.txt"
    $NewReq = Join-Path $ProjectPath "requirements.txt"

    if (Test-Path -LiteralPath $OldReq) {
        if (Test-Path -LiteralPath $NewReq) {
            $oldHash = (Get-FileHash -LiteralPath $OldReq -Algorithm SHA256).Hash
            $newHash = (Get-FileHash -LiteralPath $NewReq -Algorithm SHA256).Hash

            if ($oldHash -ne $newHash) {
                throw "Both requirements files exist with different contents in $ProjectName. Nothing was overwritten."
            }

            Remove-Item -LiteralPath $OldReq -Force
            Ok "${ProjectName}: duplicate setup\requirements.txt removed; root requirements.txt preserved."
        }
        else {
            Move-Item -LiteralPath $OldReq -Destination $NewReq
            Ok "${ProjectName}: requirements.txt moved to project root."
        }
    }
    elseif (Test-Path -LiteralPath $NewReq) {
        Ok "${ProjectName}: requirements.txt already at project root."
    }
    else {
        if ($ProjectName -eq "P2") {
            New-Item -ItemType File -Path $NewReq -Force | Out-Null
            Ok "P2: empty root requirements.txt created."
        }
        elseif ($ProjectName -eq "P1") {
            throw "P1 requirements.txt was not found. It should exist because P1 has a populated environment."
        }
        else {
            Info "${ProjectName}: no requirements.txt currently needed; project remains document-only."
        }
    }

    if (Test-Path -LiteralPath $SetupDir) {
        $remaining = @(Get-ChildItem -LiteralPath $SetupDir -Force)

        if ($remaining.Count -eq 0) {
            Remove-Item -LiteralPath $SetupDir -Force
            Ok "${ProjectName}: empty setup folder removed."
        }
        else {
            Info "${ProjectName}: setup folder retained because it still contains other files."
        }
    }
}

Step "2/4 - REBUILD P2 .venv AS A MINIMAL ENVIRONMENT"

$P2 = Join-Path $ProjectsRoot "P2"
$P2Venv = Join-Path $P2 ".venv"
$P2Req = Join-Path $P2 "requirements.txt"

if (Test-Path -LiteralPath $P2Venv) {
    Remove-Item -LiteralPath $P2Venv -Recurse -Force
    Ok "Previous P2\.venv removed."
}

$PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($null -eq $PyLauncher) {
    throw "py.exe was not found in PATH."
}

& $PyLauncher.Source -3.12 -m venv $P2Venv
if ($LASTEXITCODE -ne 0) {
    & $PyLauncher.Source -3 -m venv $P2Venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the P2 virtual environment."
    }
}

$P2Python = Join-Path $P2Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $P2Python)) {
    throw "P2 Python interpreter was not created: $P2Python"
}

& $P2Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip in P2."
}

if ((Get-Item -LiteralPath $P2Req).Length -ne 0) {
    throw "P2 requirements.txt is not empty. Minimal P2 initialization was requested, so no project dependencies were installed."
}

$SetupGlobal = Join-Path $PortfolioRoot "setup-global"
if (Test-Path -LiteralPath $SetupGlobal) {
    $SitePackages = (& $P2Python -c "import site; print(site.getsitepackages()[0])").Trim()
    $PthFile = Join-Path $SitePackages "portfolio_setup_global.pth"
    Set-Content -LiteralPath $PthFile -Value $SetupGlobal -Encoding UTF8
    Ok "P2: setup-global linked through a .pth file."
}

Step "3/4 - VALIDATE P1/P2 STANDARD"

$P1 = Join-Path $ProjectsRoot "P1"
$P1Req = Join-Path $P1 "requirements.txt"
$P1Venv = Join-Path $P1 ".venv"

foreach ($RequiredPath in @($P1Req, $P1Venv, $P2Req, $P2Venv)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required final path missing: $RequiredPath"
    }
    Ok $RequiredPath
}

$Freeze = @(& $P2Python -m pip list --format=freeze)

$NonBase = @(
    $Freeze |
    ForEach-Object {
        if ($_ -match '^([^=]+)==') {
            $matches[1].ToLowerInvariant()
        }
    } |
    Where-Object {
        $_ -and $_ -notin @("pip", "setuptools", "wheel")
    }
)

if ($NonBase.Count -gt 0) {
    throw "P2 .venv is not minimal. Unexpected packages: $($NonBase -join ', ')"
}

Ok "P2 .venv is minimal (no project dependencies installed)."

Step "4/4 - FINAL STRUCTURE"

Write-Host ""
Write-Host "P1" -ForegroundColor White
Write-Host "  .venv\"
Write-Host "  requirements.txt"
Write-Host "  scripts\"
Write-Host "  data\"
Write-Host "  outputs\"
Write-Host "  results\"

Write-Host ""
Write-Host "P2" -ForegroundColor White
Write-Host "  .venv\"
Write-Host "  requirements.txt   (empty by design)"
Write-Host "  scripts\"
Write-Host "  data\"
Write-Host "  outputs\"
Write-Host "  results\"

Write-Host ""
Write-Host "P3 / P4" -ForegroundColor White
Write-Host "  README.md"
Write-Host "  results\"
Write-Host "  (no .venv / requirements.txt unless they become executable projects)"

Write-Host ""
Write-Host "STANDARDIZATION COMPLETE" -ForegroundColor Green
Write-Host ""
Write-Host "Activate P1:"
Write-Host 'cd "C:\Users\vitor\Documents\Portfolio\Projects\P1"'
Write-Host '.\.venv\Scripts\Activate.ps1'
Write-Host ""
Write-Host "Activate P2:"
Write-Host 'cd "C:\Users\vitor\Documents\Portfolio\Projects\P2"'
Write-Host '.\.venv\Scripts\Activate.ps1'
