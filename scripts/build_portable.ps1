[CmdletBinding()]
param(
    [string]$SourceDirectory = "",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $SourceDirectory) {
    $SourceDirectory = Join-Path $projectRoot "dist-fixed\bioeng-qpcr"
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot "dist-release"
}

$source = Get-Item -LiteralPath $SourceDirectory
$executable = Join-Path $source.FullName "bioeng-qpcr.exe"
$internal = Join-Path $source.FullName "_internal"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Packaged executable not found: $executable"
}
if (-not (Test-Path -LiteralPath $internal -PathType Container)) {
    throw "Packaged _internal directory not found: $internal"
}

$versionLine = Select-String -LiteralPath (Join-Path $projectRoot "pyproject.toml") -Pattern '^version = "([^"]+)"$' | Select-Object -First 1
if (-not $versionLine) {
    throw "Could not determine the project version from pyproject.toml."
}
$version = $versionLine.Matches[0].Groups[1].Value
$releaseName = "bioeng-qpcr-$version-win64-portable"

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$output = (Get-Item -LiteralPath $OutputDirectory).FullName
$stage = Join-Path $output $releaseName
$archive = Join-Path $output "$releaseName.zip"
$archiveChecksum = "$archive.sha256"
foreach ($target in ($stage, $archive, $archiveChecksum)) {
    if (Test-Path -LiteralPath $target) {
        throw "Release target already exists; move or remove it before rebuilding: $target"
    }
}

$smoke = Start-Process -FilePath $executable -ArgumentList "--smoke-test" -PassThru -Wait -WindowStyle Hidden
if ($smoke.ExitCode -ne 0) {
    throw "Packaged scientific smoke test failed with exit code $($smoke.ExitCode)."
}

Copy-Item -LiteralPath $source.FullName -Destination $stage -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot "packaging\README-PORTABLE.txt") -Destination (Join-Path $stage "README-PORTABLE.txt")

$manifestPath = Join-Path $stage "MANIFEST.sha256"
$manifestLines = Get-ChildItem -LiteralPath $stage -File -Recurse |
    Sort-Object FullName |
    ForEach-Object {
        $relative = [System.IO.Path]::GetRelativePath($stage, $_.FullName).Replace('\', '/')
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash *$relative"
    }
[System.IO.File]::WriteAllLines($manifestPath, $manifestLines, [System.Text.UTF8Encoding]::new($false))

Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
$zipHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText(
    $archiveChecksum,
    "$zipHash *$releaseName.zip`n",
    [System.Text.UTF8Encoding]::new($false)
)

$signature = Get-AuthenticodeSignature -LiteralPath $executable
$archiveFile = Get-Item -LiteralPath $archive
[pscustomobject]@{
    Version = $version
    SmokeTestExitCode = $smoke.ExitCode
    Archive = $archiveFile.FullName
    ArchiveSizeMB = [math]::Round($archiveFile.Length / 1MB, 2)
    ArchiveSHA256 = $zipHash.ToUpperInvariant()
    ExecutableSignature = $signature.Status
}

